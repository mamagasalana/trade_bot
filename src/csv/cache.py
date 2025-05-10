import os
import pickle
import logging
import hashlib

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


class CACHE2:
    def __init__(self, version_name):
        self.folder = os.path.join('/home/ytee3/caches', version_name)
        os.makedirs(self.folder, exist_ok=True)

    def _hash_key(self, key):
        """Convert the key into a safe hashed filename."""
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def _file_path(self, key):
        """Get the full file path for a given key."""
        return os.path.join(self.folder, self._hash_key(key))

    def __getitem__(self, key):
        filename = self._file_path(key)
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as ifile:
                    logging.debug(f"Loading pickle from {filename}")
                    return pickle.load(ifile)
            except (pickle.PickleError, EOFError, Exception) as e:
                logging.debug(f"Error loading pickle file {filename}: {e}")
                return None
        logging.debug(f"Pickle file {filename} not found.")
        return None

    def __setitem__(self, key, value):
        filename = self._file_path(key)
        try:
            with open(filename, 'wb') as ofile:
                pickle.dump(value, ofile)
                logging.debug(f"Saved pickle to {filename}")
        except Exception as e:
            logging.debug(f"Error saving pickle to {filename}: {e}")

    def __contains__(self, key):
        return os.path.exists(self._file_path(key))