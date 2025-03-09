import pandas as pd

class Slicer:
    def __init__(self, data : pd.DataFrame, mode):
        self.data = data.to_numpy()
        self.mode = mode

        self.slicing_mode_list = {
                    'fix_start': "index <= {i}",
                    'fix_end': "index >= {i}",
                    'rolling10': "index >= {i} - 10*260 + 1 and index <= {i}",
                    'rolling15': "index >= {i} - 15*260 + 1 and index <= {i}",
                    'rolling5': "index >= {i} - 5*260 + 1 and index <= {i}",
                }
        modes = list(self.slicing_mode_list.keys())
        assert mode in modes, f'mode must be one of {modes}'
    
    def get(self, i):
        if self.mode == 'fix_start':
            return self.data[:i+1]
        elif self.mode == 'fix_end':
            return self.data[i:]
        elif self.mode == 'rolling10':
            return self.data[max(0, i- 10*260 +1) : i+1]
        elif self.mode == 'rolling5':
            return self.data[max(0, i- 5*260 +1) : i+1]
        elif self.mode == 'rolling15':
            return self.data[max(0, i- 15*260 +1) : i+1]