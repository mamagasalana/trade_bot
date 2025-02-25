import os
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class custom_PCA:
    def __init__(self):
        self.base_path = "files/pca/pca"
        self.extension = '.csv'

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
    
    def get_pca_components(self, chart=False, min_variance=0.05):
        scaler = StandardScaler()
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
        import matplotlib.pyplot as plt
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
        
        df.to_csv(self.get_next_version())
        return df