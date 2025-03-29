import os
import pickle
import logging

class CACHE:
    def __init__(self, filename):
        self.folder = os.path.join('files', 'cache')
        self.filename = filename
        os.makedirs(self.folder, exist_ok=True)

    def get(self, filename=None):
        if filename is None:
            filename = self.filename
        filename = os.path.basename(filename)  # Prevent path traversal
        return os.path.join(self.folder, filename)

    def get_pickle(self, filename=None):
        if filename is None:
            filename = self.filename
        f = self.get(filename)
        if os.path.exists(f):
            try:
                with open(f, 'rb') as ifile:
                    logging.debug(f"Loading pickle from {f}")
                    return pickle.load(ifile)
            except (pickle.PickleError, EOFError, Exception) as e:
                logging.debug(f"Error loading pickle file {f}: {e}")
                return None
        logging.debug(f"Pickle file {f} not found.")
        return None
    
    def set_pickle(self, data, filename=None):
        if filename is None:
            filename = self.filename
        f = self.get(filename)
        try:
            with open(f, 'wb') as ofile:
                pickle.dump(data, ofile)
                logging.debug(f"Pickle file {f} successfully saved.")
        except Exception as e:
            logging.debug(f"Error saving pickle file {f}: {e}")
