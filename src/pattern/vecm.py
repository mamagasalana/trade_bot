import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import statsmodels.tsa.vector_ar.vecm as vecm
from sklearn.preprocessing import StandardScaler
import itertools
from src.pattern.pca import custom_PCA
from  src.csv.fred  import FRED
from src.csv.reader import reader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
from functools import partial
from src.csv.slicer import Slicer
from tqdm import tqdm
import multiprocessing as mp
import pickle
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
import json

class VECM:
    def __init__(self):
        self.fr = FRED()
        self.debug=True
        #365 - 52*2 = 261 days (after excluded weekend)
        self.custom_pca = custom_PCA()

        # self.reader = reader()

    def compute_historical_vecm_pca(self, pairs: list, 
                                mode: str, 
                                qry: str=None, 
                                deterministic= "ci",
                                version:int=1):
        self.debug=False
        pca_version = self.custom_pca.get_version(version=version,base=True)
        if qry is None:
            filename = f'files/cointegration/{pca_version}_{"_".join(sorted(pairs))}_{mode}_{deterministic}.pkl'
        else:
            filename = f'files/cointegration/{pca_version}_{"_".join(sorted(pairs))}_{mode}_{deterministic}_{qry}.pkl'

        if os.path.exists(filename):
            data2= pd.read_pickle(filename)
        else:
            data = self.fr.get_pairs(pairs)
            if qry is not None:
                data = data.query(qry)

            data_pca = pd.read_csv(self.custom_pca.get_version(version=version))
            data = data.merge(data_pca, how='inner', left_index=True, right_on='datetime')
            data = data.set_index('datetime') 
            data2= data.reset_index(drop=True)

            slicer = Slicer(data2, mode)

            results = data2.index.to_series().parallel_apply(lambda i: 
                self.vecm_model(slicer.get(i), deterministic=deterministic))
                
            data2['vecm'] = results
            data2.index = data.index
            data2.to_pickle(filename)
        
        return data2
    
    
    def compute_historical_vecm(self, pairs: list, 
                                mode: str, 
                                qry: str=None, 
                                deterministic= "ci"):
        self.debug=False
        if qry is None:
            filename = f'files/cointegration/{"_".join(sorted(pairs))}_{mode}_{deterministic}.pkl'
        else:
            filename = f'files/cointegration/{"_".join(sorted(pairs))}_{mode}_{deterministic}_{qry}.pkl'

        if os.path.exists(filename):
            data2= pd.read_pickle(filename)
        else:
            data = self.fr.get_pairs(pairs)
            if qry is not None:
                data = data.query(qry)
            data2= data.reset_index(drop=True)

            slicer = Slicer(data2, mode)

            results = data2.index.to_series().parallel_apply(lambda i: 
                self.vecm_model(slicer.get(i), deterministic=deterministic))
                
            data2['vecm'] = results
            data2.index = data.index
            data2.to_pickle(filename)
        
        return data2
    
    def plot_spread(self, data, key, title=f'Actual vs. Spread'):
        fig, ax1 = plt.subplots(figsize=(12, 4))

        # Plot actual and modeled oil prices on the primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel(f'{key}', color=color)
        data.plot(y=key, color=['blue', 'orange'], ax=ax1, legend=False)
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for the ECT
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('spread', color=color)  # we already handled the x-label with ax1
        data.plot(y='spread', color='green', ax=ax2, legend=False, linestyle=':')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
        ax2.axhline(-2, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0
        ax2.axhline(2, color='gray', linewidth=0.8, linestyle='--')  # Adding a horizontal line at y=0

        # Adding a title and legend
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(title)
        fig.legend(loc="upper left", bbox_to_anchor=(1.02,1), bbox_transform=ax1.transAxes)
        plt.show()

    def plot_historical_vecm(self, pairs: list, 
                             mode: str, 
                             mode2: str=None, 
                             qry : str=None,
                             deterministic= "ci", 
                             raw=False):
        data = self.compute_historical_vecm(pairs=pairs, mode=mode, qry=qry, deterministic=deterministic)
        data2 = data.reset_index(drop=True)
        if mode2 is None:
            mode2 = mode
        slicer = Slicer(data2[pairs], mode2)
        spread = data2.index.to_series().apply(lambda i: 
                    self.get_scaled_spread(slicer.get(i), data2.iloc[i]['vecm'] ))
        spread.index= data.index
        data['spread'] = spread
        projected =  data2.index.to_series().apply(lambda i: 
                    self.get_projected(slicer.get(i), data2.iloc[i]['vecm'] ))
        projected.index = data.index
        has_projected = not projected[~projected.isna()].empty
        if has_projected:
            data[['projected_%s' % x for x in pairs]] = projected.apply(pd.Series)
        if raw:
            return data
        
        for key in pairs:
            if has_projected:
                self.plot_spread(data, [key, 'projected_%s' % key])
            else:
                self.plot_spread(data, key)

    def plot_historical_vecm_v2(self, folder):
        config = json.load(open(os.path.join(folder,  'config.json'), 'r'))
        pairs= config['pairs']
        mode = config['slicing_mode']
        print(f'{pairs} {mode}')
        with open(os.path.join(folder,  "vecm_results.pkl"), "rb") as f:
            vecm_results_dict = pickle.load(f)
        df = pd.read_csv(os.path.join(folder,  'datafile.csv'), index_col=[0, 1])
        
        unique_blocks = df.index.get_level_values(0).unique()
        blocks_series2 = pd.Series(unique_blocks)
        spread = blocks_series2.apply(lambda i: self.get_scaled_spread(df.xs(i, level=0).to_numpy(), vecm_results_dict.get(i)))
        projected =  blocks_series2.apply(lambda i: self.get_projected(df.xs(i, level=0).to_numpy(), vecm_results_dict.get(i)))

        fx = self.fr.get_pairs(pairs)
        projected_array = np.vstack(projected.to_numpy())  # Converts list-like rows into a 2D NumPy array
        spread_array = spread.to_numpy().reshape(-1, 1)  # Convert spread to column vector (3298,1)

        # Now we can hstack without issues
        result = np.hstack([projected_array, spread_array])
        newidx = df.index.to_frame(index=False).groupby(0)[1].last()
        projected_fx = pd.DataFrame(result , index=newidx, columns=['projected_%s' % x for x in df.columns]+['spread'])

        fx2 = fx.merge(projected_fx, how='right', left_index=True, right_index=True)
        for key in pairs:
            upper_bound = fx2[key].max()*1.02
            lower_bound = fx2[key].min() *0.98
            # Clamp the projected value between these bounds
            fx2['projected_%s' % key] = fx2['projected_%s' % key].clip(lower=lower_bound, upper=upper_bound)
            self.plot_spread(fx2, [key, 'projected_%s' % key], title=f'{"_".join(pairs)} {mode}')

    def get_data(self, raw=False):
        df = self.fr.get_tag('daily')
        if raw:
            return df
        
        farb_idx = df[(df.eclass=='Factors Affecting Reserve Balances')].event.unique()
        irs_idx = df[(df.eclass=='Interest Rate Spreads')].event.unique()
        cp_idx = df[df.eclass=='Commercial Paper'].event.unique()


        pivot_df = df.pivot(index='datetime', columns='event', values='actual').dropna(axis=1, thresh=10)#.ffill().dropna()
        pivot_df[farb_idx] = pivot_df[farb_idx].fillna(0)
        pivot_df[irs_idx] = pivot_df[irs_idx].ffill()
        pivot_df[cp_idx] = pivot_df[cp_idx].ffill()
        pivot_df = pivot_df.ffill().dropna()

        return pivot_df
    

    def unit_root_test(self, col, level=0, stats=False):
        adf_result = adfuller(col, autolag='AIC')  # Automatically select the lag based on AIC

        # Interpretation of results
        if adf_result[1] < 0.05:
            if stats:
                # Output the results
                print('ADF Statistic:', adf_result[0])
                print('p-value:', adf_result[1])
                print('Critical Values:')
                for key, value in adf_result[4].items():
                    print(f'\t{key}: {value}')
            print(f"The series L({level}) is likely stationary (reject the null hypothesis of unit root).")
        else:
            self.unit_root_test(col.diff().dropna(), level=level+1, stats=stats)
            # print("The series is likely non-stationary (fail to reject the null hypothesis of unit root).")

    def select_order(self, data, raw=False):
        model = VAR(data)
        results = model.select_order(maxlags=15)  # You can adjust 'maxlags' as necessary
        self.custom_print(results.summary())

        # why -1
        # because VAR order shows lag
        # but VECM require lag difference
        # Each difference Δy_t = y_t - y_t-1 reduces the number of data points available for analysis by one for each lag level.
        # This reduction occurs because the difference calculation combines information from two consecutive periods into one difference value.
        # Here, Δy_t represents the change from one period to the next (y_t - y_t-1), capturing the period-to-period changes in the data.
        if raw:
            return results
        else:
            return results.bic - 1

    def custom_print(self, val):
        if self.debug:
            print(val)

    def cointegration_test(self, data, det_order=1, k_ar_diff=2):
        # Assuming 'data' is your DataFrame and 'maxlags' is defined
        johansen_test = vecm.coint_johansen(data, det_order=0, k_ar_diff=2)

        self.custom_print("\nJohansen Cointegration Test Results:")
        self.custom_print("=====================================")
        # self.custom_print(f"Trace Statistics: {johansen_test.lr1}" )
        # self.custom_print(f"Critical Values (90%, 95%, 99%): {johansen_test.cvt}")
        # self.custom_print(f"Eigen Statistics: {johansen_test.lr2}", )
        # self.custom_print(f"Max-Eigen Critical Values (90%, 95%, 99%): {johansen_test.cvm}")
        # self.custom_print(f"Eigenvalues: {johansen_test.eig}", )
        num_of_cointegrating =  0
        # Interpretation
        self.custom_print("\nInterpretation:")
        for i, (trace_stat, cv_95) in enumerate(zip(johansen_test.lr1, johansen_test.cvt[:, 1])):

            if trace_stat > cv_95:
                
                num_of_cointegrating = i+1
                continue
                # self.custom_print(f"r <= {i}: Trace Statistic = {trace_stat:.4f}, 95% Critical Value = {cv_95:.4f}")
                # self.custom_print(f"  => Reject null hypothesis of r <= {i}, suggesting at least {i+1} cointegrating relations at 95% confidence level.")

            else:
                self.custom_print(f"r <= {i}: Trace Statistic = {trace_stat:.4f}, 95% Critical Value = {cv_95:.4f}")
                self.custom_print(f"  => Fail to reject null hypothesis of r <= {i}, suggesting no more than {i} cointegrating relations at 95% confidence level.")
                num_of_cointegrating = i
                break
        self.custom_print("=====================================\n\n")
        return num_of_cointegrating, johansen_test

    def vecm_model(self, data, det_order=0, deterministic='ci', visualize=False):

        if data.shape[0] < 1000:
            return None
        
        k_ar_diff = self.select_order(data)
        num_of_cointegration, _ = self.cointegration_test(data, det_order=det_order, k_ar_diff=k_ar_diff)

        self.custom_print(f'k_ar_diff: {k_ar_diff}, num_of_cointegration: {num_of_cointegration}')
        # assert  num_of_cointegration, "No cointegration"
        if not num_of_cointegration:
            return None

        model = vecm.VECM(data, k_ar_diff=k_ar_diff, coint_rank=num_of_cointegration, deterministic=deterministic)
        vecm_result = model.fit()
        self.custom_print(vecm_result.summary())
        
        if visualize:
            pairs = data.columns
            spread_scaled = self.get_scaled_spread(data.to_numpy(), vecm_result, raw=True)
            projected = self.get_projected(data.to_numpy(), vecm_result, raw=True)
            data2 = data.copy()
            data2['spread'] = spread_scaled
            data2[['projected_%s' % x for x in pairs]] = projected
            if spread_scaled is not None:
                for col in pairs:
                    upper_bound = data2[col].max()*1.02
                    lower_bound = data2[col].min() *0.98
                    # Clamp the projected value between these bounds
                    data2['projected_%s' % col] = data2['projected_%s' % col].clip(lower=lower_bound, upper=upper_bound)
                    self.plot_spread(data2, [col, 'projected_%s' % col])

        return vecm_result
    
        # # Calculating RMSE and MAE from the point where modeled data begins
        # rmse = np.sqrt(mean_squared_error(data['Oil'], modeled_oil))
        # mae = mean_absolute_error(data['Oil'], modeled_oil)

        # print(f"RMSE: {rmse}")
        # print(f"MAE: {mae}")

    def get_projected(self, data, vecm_result: vecm.VECMResults, raw=False):
        # general equation to express x in terms of y
        # xₜ = - ( (∑₍ᵢ₌₁₎ʳ aᵢ · (Bᵢ ⋅ yₜ)) + (∑₍ᵢ₌₁₎ʳ aᵢ · cᵢ) ) / (∑₍ᵢ₌₁₎ʳ aᵢ · bᵢ)
        if vecm_result is None:
            if not raw:
                return np.full(data.shape[1], np.nan)
            return

        out = []
        for j in range(data.shape[-1]):
            beta = vecm_result.beta            # shape: (n, r)
            c_vec = vecm_result.det_coef_coint # shape: (r,)
            a_j = vecm_result.alpha[j, :]      # adjustment coefficients for equation j, shape: (r,)

            # Denom: D = ∑_{i=1}^{r} a[j,i] * beta[j, i]
            if c_vec.shape[0] ==0:
                sum_ac = 0 #no intercept
            else:
                sum_ac = np.dot(a_j, c_vec.T)  #to achieve (∑₍ᵢ₌₁₎ʳ aᵢ · cᵢ) )
            D = np.dot(a_j, beta[j, :])
            sum_aby =np.zeros(data.shape[0])
            n = beta.shape[0]
            for k in range(n):
                if k != j:
                    sum_aby += data[:, k] *np.dot(a_j, beta[k, :])


            ret = -(sum_aby + sum_ac )/ D
            out.append(ret)
        project = np.column_stack(out)
        if raw:
            return project
        else:
            return project[-1]

    def get_scaled_spread(self, data, vecm_result: vecm.VECMResults, raw=False):

        if vecm_result is None:
            return
        projected = self.get_projected(data, vecm_result, raw=True)
        spread =  data[:, 0] - projected[:, 0]
        scaler = StandardScaler()
        spread_scaled = scaler.fit_transform(spread.reshape(-1, 1))
        # ect_data_scaled = ect_data
        
        if raw:
            return spread_scaled
        else:
            return spread_scaled.reshape(-1)[-1]
        

if __name__ == '__main__':
    v =VECM()
    v.compute_historical_vecm_pca(['CADCHF', 'EURCHF'], 'fix_start')