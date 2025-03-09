import os
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.csv.slicer import Slicer
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
import json

class custom_PCA:
    def __init__(self):
        self.base_path = "files/pca/pca"
        self.extension = '.csv'
        self.scaler = StandardScaler()
        self.pca = None

    def get_next_version(self, base=False):
        """ 
        Find the next available version number for a file in the given directory. 
        """
        version = 1
        while os.path.exists(f"{self.base_path}{version}{self.extension}"):
            version += 1

        ret = f"{self.base_path}{version}{self.extension}"
        if base:
            return os.path.basename(ret).replace(self.extension, '')
        else:
            return ret

    def get_latest_version(self, base=False):
        """
        Find the latest version number of files in the given directory.
        Returns the latest version number or 0 if no files are found.
        """
        version = 1
        while os.path.exists(f"{self.base_path}{version}{self.extension}"):
            version += 1

        ret =  f"{self.base_path}{version-1}{self.extension}"
        if base:
            return os.path.basename(ret).replace(self.extension, '')
        else:
            return ret
        
    def get_version(self, version:int, base=False):
        ret = f"{self.base_path}{version}{self.extension}"
        if base:
            return os.path.basename(ret).replace(self.extension, '')
        else: 
            return ret
    
    def get_pca_components(self, data=None, chart=False, min_variance=0.05, save_flag= False):
        scaler = StandardScaler()

        if data is None:
            save_flag= True
            fname =f'files/pca/pca_raw.parquet'
            assert os.path.exists(fname), "src/csv/Please run pca_raw.py"
            data=  pd.read_parquet(fname)

        print(f"Data shape before {data.shape}" )
        data =data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        print(f"Data shape after {data.shape}" )
        X_scaled = scaler.fit_transform(data)

        pca = PCA()
        principalComponents = pca.fit_transform(X_scaled)

        # To see the variance explained by each component
        explained_variance = pca.explained_variance_ratio_
        explained_variance2 = explained_variance[:40]

        if chart:
            # Create a scree plot
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(explained_variance2) + 1), explained_variance2, marker='o', linestyle='-')
            plt.title('Scree Plot of PCA')
            plt.xlabel('Principal Component')
            plt.ylabel('Variance Explained')
            plt.xticks(range(1, len(explained_variance2) + 1))  # Ensure x-axis labels match the number of components
            plt.grid(True)
            plt.show()

        n_components = np.where(explained_variance>min_variance)[0][-1]+1

        cumsum = explained_variance[:n_components].sum()*100
        print("this explained %.2f%% of variance" % cumsum)
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(X_scaled)

        df =  pd.DataFrame(data = principalComponents,
                            columns = [f'principal_component_{i}' for i in range(n_components)],
                            index=data.index
                            )
        if save_flag:
            df.to_csv(self.get_next_version())
        return df
    
    def get_pca_components_v2(self, data, mode, n_components=3, metadata=""):

        """
        create a pca that refits every day
        """

        print(f"Data shape before {data.shape}" )
        data = data.replace([np.inf, -np.inf], np.nan).dropna(thresh=len(data)*0.98, axis=1).dropna()
        print(f"Data shape after {data.shape}" )
        self.pca = PCA(n_components=n_components)

        data2= data.reset_index(drop=True)
        self.slicer = Slicer(data2, mode)
        principalComponents = data2.index.to_series().parallel_apply(lambda i:  self._get_pca_components_v2_unit_func(i))

        dfs = []
        for i, arr in enumerate(principalComponents):
            if arr is None:
                continue
            else:
                n_rows = arr.shape[0]
                # Create a MultiIndex: first level is row number, second is the index i of the array
                index = pd.MultiIndex.from_arrays(
                    [data.index[:n_rows], np.full(n_rows, i)],
                    names=['row', 'block']
                )
                # Create a DataFrame from the array with columns names for clarity
            df_temp = pd.DataFrame(arr, index=index, columns = [f'principal_component_{i}' for i in range(n_components)])
            dfs.append(df_temp)
        df_all = pd.concat(dfs)

        fname = self.get_next_version()
        df_all.to_csv(fname)
        print("File save as %s" % fname)
        self.pca_metadata(fname, metadata)
        return df_all
    
    def pca_metadata(self, fname, metadata):
        json_file_path = self.base_path + '.json'

        data = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)

        # Append the new entry
        data.append({
            'fname' : fname,
            'metadata' : metadata
        })

        # Write the updated data back to the JSON file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_pca_components_v2_unit_func(self, i):
        if i < 1000:
            return None

        data = self.slicer.get(i)
        X_scaled = self.scaler.fit_transform(data)
        principalComponents = self.pca.fit_transform(X_scaled)
        return principalComponents
    
if __name__ == '__main__':
    import pandas as pd
    from src.pattern.vecm import VECM
    from src.csv.bond import BONDS_investingcom
    v = VECM()


    pairs = ['USDCHF', 'USDCAD']
    unitpairs = [ z  for y in [[x[:3], x[3:]] for x in pairs] for z in y]
    unitpairs = list(sorted(set(unitpairs)))
    rate_differentials = BONDS_investingcom().get_diff(unitpairs)

    pca = custom_PCA()
    data =  v.get_pairs(pairs)
    # data2 =data.copy()
    data_pca = pd.read_csv('files/pca/pca1.csv')
    data2 = data.merge(data_pca, how='inner', left_index=True, right_on='datetime')
    data2 = data2.set_index('datetime') 
    rd =rate_differentials[rate_differentials.index.isin(data2.index)].copy()


    rd3 = pca.get_pca_components_v2(data=rd, mode='fix_start', metadata=f'investing.com_{pairs}_rd =rate_differentials[rate_differentials.index.isin(data2.index)].copy()')