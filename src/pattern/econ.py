import datetime
import pandas as pd
import os
import logging
import numpy as np
from global_macro_data import gmd
from src.csv.cache import CACHE2
import pymc as pm
import pytensor.tensor as at   
import arviz as az

# source
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5349012

my_cache =  CACHE2('cftc.cache')

class ECON:
    def __init__(self):
        self.df = self.get_df()
        self.df = self.df[(self.df.year >2000) & (self.df.year<2025)].copy()

        self.ISO_COL   = "ISO3"
        self.TIME_COL  = "year"
        self.FX_COL    = "USDfx"          # <- change to your currency level column
        # self.ECON_COLS = ['nGDP','rGDP',
        #     'rGDP_pc', 'deflator', 'cons', 'cons_GDP',
        #     'inv', 'inv_GDP', 'finv', 'finv_GDP',  'exports',
        #     'exports_GDP',  'imports', 'imports_GDP', 
        #     'CA', 'CA_GDP',  'REER', 'govexp', 'gen_govexp',
        #     'gen_govexp_GDP', 'cgovexp', 'cgovexp_GDP', 'govrev', 'gen_govrev',
        #     'cgovrev', 'gen_govrev_GDP', 'cgovrev_GDP', 'govtax', 'gen_govtax',
        #     'cgovtax', 'gen_govtax_GDP', 'cgovtax_GDP', 'govdef_GDP',
        #     'gen_govdef_GDP', 'gen_govdef', 'cgovdef_GDP', 'cgovdef', 'govdebt_GDP',
        #     'gen_govdebt_GDP', 'gen_govdebt', 'cgovdebt_GDP', 'cgovdebt', 'HPI',
        #     'CPI', 'infl', 'pop', 'unemp', 'strate', 'ltrate', 'cbrate', 'M0', 'M1',
        #     'M2', 'M3', ]
        self.LOGDIFF_COLS = [             # strictly-positive scale vars (use log-diff)
            "rGDP_pc",
        ]
        self.DIFF_COLS = [                # signed/ratio/rate vars (use plain diff or asinh-diff)
            "strate","ltrate","infl",'rGDP', "CA_GDP","gen_govdebt_GDP","gen_govdef_GDP", "gen_govexp_GDP", "unemp"]
        self.LEVEL_COLS = [               # keep levels (optionally z-score within year)
            "REER"
        ]

        
        self.ISO_KEEP  = ["USA","CAN","JPN","NZL","AUS","CHE"]          # optional country filter
        self.PREDICT_NEXT_PERIOD = False  # set False if you want same-year cross-section prediction

        if self.ISO_KEEP:
            self.df = self.df[self.df[self.ISO_COL].isin(self.ISO_KEEP)]
        

        (X_tr, y_tr), (X_va, y_va), (X_te, y_te), X_cols, df_model = self.make_dataset()
        self.df2 = df_model[ ['year', 'ISO3', self.FX_COL+'_ld_z']+X_cols ]

    @my_cache
    def get_df(self):
        return gmd()
    

    # --- 1) helper: log-diff by country (sorted by self.TIME_COL) ---
    def logdiff_by_country(self, df, cols):
        df = df.sort_values([self.ISO_COL, self.TIME_COL]).copy()
        for c in cols:
            df[c + "_ld"] = df.groupby(self.ISO_COL, sort=False, group_keys=False)[c].apply(lambda s: np.log(s).diff())

                
        return df
    
    def diff_by_country(self, df, cols):
        df = df.sort_values([self.ISO_COL, self.TIME_COL]).copy()
        for c in cols:
            df[c + "_d"] = df.groupby(self.ISO_COL, sort=False, group_keys=False)[c].apply(lambda s: s.diff())

                
        return df
    

    # --- 2) helper: cross-sectional z-score within each year ---
    def cross_sectional_z(self, df, cols):
        # z = (x - mean_year) / std_year evaluated across all ISO within that year
        g = df.groupby(self.TIME_COL)
        means = g[cols].transform("mean")
        stds  = g[cols].transform("std")

        z = (df[cols] - means) / stds
        z.columns = [c + "_z" for c in cols]
        return pd.concat([df, z], axis=1)

    # --- 3) Build dataset ---
    def make_dataset(self):
        # df = self.df[(self.df.year >2000) & (self.df.year<2025)].copy()
        df =self.df.copy()



        # make log-diffs for econ features and FX
        df = self.logdiff_by_country(df, self.LOGDIFF_COLS + [self.FX_COL])
        df = self.diff_by_country(df, self.DIFF_COLS)

        # cross-sectional z-score by year for econ ld features and FX ld target
        cols1 = [c + "_ld" for c in self.LOGDIFF_COLS]
        cols2 = [c + "_d" for c in self.DIFF_COLS]
        cols3 = self.LEVEL_COLS
        target_ld_col = self.FX_COL + "_ld"

        df = self.cross_sectional_z(df, cols1 + cols2 + cols3 + [target_ld_col])

        # choose X features: the z-scored econ log-diffs
        X_cols = [c + "_z" for c in cols1+cols2+cols3]

        # target: z-scored FX log-diff
        y_col = target_ld_col + "_z"

        # If predicting NEXT period's FX move, shift the target forward within each ISO
        if self.PREDICT_NEXT_PERIOD:
            df[y_col] = df.groupby(self.ISO_COL, sort=False)[y_col].shift(-1)

        # drop rows where we don't have full features or target
        df_model = df.dropna(subset=X_cols + [y_col]).copy()

        # For a time-aware split, pick by year (example: train<=2016, val=2017-2020, test>=2021)
        # Adjust cutoffs for your panel’s range
        train = df_model[df_model[self.TIME_COL] <= 2016]
        val   = df_model[(df_model[self.TIME_COL] >= 2017) & (df_model[self.TIME_COL] <= 2020)]
        test  = df_model[df_model[self.TIME_COL] >= 2021]

        X_tr, y_tr = train[X_cols].values, train[y_col].values
        X_va, y_va = val[X_cols].values,   val[y_col].values
        X_te, y_te = test[X_cols].values,  test[y_col].values

        return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), X_cols, df_model

    def plot(self, key):
        iso3 = ["USA", "CAN", "JPN", "NZL", "AUS", "CHE"]


        df_wide = (
            self.df[(self.df["ISO3"].isin(iso3)) & (self.df["year"] > 2000) & (self.df["year"] < 2025)]
            .pivot_table(index="year", columns="ISO3", values=key, aggfunc="last")
            .sort_index()
            # .reindex(columns=iso3)  # optional: enforce column order
        )

        # 1) Period-to-period log differences (≈ growth rates)
        dlog = np.log(df_wide).diff()          # in log units
        dlog_pct = 100 * dlog                   # ≈ % growth per period

        # 2) Rebase each country so the first non-missing observation = 100
        base = df_wide.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        rebased = (df_wide / base) * 100

        # If you specifically want a "log-index" rebased to 100 (i.e., exp of cum log changes):
        log_levels = np.log(df_wide)
        log_idx_100 = np.exp(log_levels.sub(log_levels.apply(lambda s: s.dropna().iloc[0]), axis=1)) * 100
        log_idx_100.plot()
        
    def preprocessing(self):
        date_col = 'year'
        id_col   = 'ISO3'

        # 1) Build the target: next-period return per currency (ex-ante)
        X_cols   = self.df2.columns[3:].tolist()
        df = self.df2.sort_values([id_col, date_col]).copy()
        df['y_next'] = df['USDfx_ld_z']
        panel = df.dropna(subset=['y_next'] + X_cols).copy()
        X = panel[X_cols].to_numpy()
        X = (X - X.mean(0)) / X.std(0)
        y = panel['y_next'].to_numpy()
        return X, y, X_cols

    def bayesian_sdf(self):
        X, y, X_cols = self.preprocessing()

        K = X.shape[1]  # number of factors (22)

        with pm.Model() as sdf_model:
            # --- Hyperpriors ---
            # Prior inclusion probability (global sparsity level)
            pi = pm.Beta('pi', alpha=1.0, beta=1.0)  # uniform on [0,1]
            
            # Spike (very small variance) and slab (diffuse) std devs
            sigma_spike = pm.Exponential('sigma_spike', lam=50.0)  # usually tiny
            sigma_slab  = pm.HalfNormal('sigma_slab', sigma=1.0)   # wider

            # Inclusion indicators (0/1) for each factor
            z = pm.Bernoulli('z', p=pi, shape=K)

            # Continuous spike-and-slab prior:
            # beta_k ~ N(0, sigma_k^2) with sigma_k = z_k*sigma_slab + (1-z_k)*sigma_spike
            sigma_beta = pm.Deterministic('sigma_beta', z * sigma_slab + (1 - z) * sigma_spike)
            coef = pm.Normal('coef', mu=0.0, sigma=sigma_beta, shape=K)

            # Intercept
            alpha = pm.Normal('alpha', mu=0.0, sigma=1.0)

            # Likelihood (homoskedastic for simplicity; cluster-robust is frequentist territory)
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            mu = alpha + at.dot(X, coef)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            idata = pm.sample(1500, tune=1500, chains=4, target_accept=0.9)
            # Derived: Posterior Inclusion Probabilities (PIPs) as E[z_k | data]
            z_post = idata.posterior['z'].stack(draws=("chain","draw")).mean('draws').values

        # ==== RESULTS =================================================================
        # Posterior means and 95% credible intervals for betas
        summary = az.summary(idata, var_names=['alpha','coef','sigma','pi','sigma_spike','sigma_slab'])
        print(summary)

        # Map factors to their PIPs and posterior mean betas
        beta_mean = idata.posterior['coef'].stack(draws=("chain","draw")).mean('draws').values
        beta_hdi  = az.hdi(idata, var_names=['coef'], hdi_prob=0.95)['coef'].values  # [K,2]

        report = pd.DataFrame({
            'factor': X_cols,
            'PIP':    z_post,
            'beta_mean': beta_mean,
            'beta_hdi_low':  beta_hdi[:,0],
            'beta_hdi_high': beta_hdi[:,1],
        }).sort_values(['PIP','beta_mean'], ascending=False)

        print("\n=== Bayesian SDF: Posterior Inclusion and Effects ===")
        print(report.to_string(index=False))
        return summary, beta_mean, beta_hdi, report
        # OPTIONAL: ex-ante forecasts (one-step-ahead) via rolling refit:
        # For production, loop over t, fit on data < t, predict at t using posterior mean beta.


if __name__ =='__main__':
    a = ECON()
