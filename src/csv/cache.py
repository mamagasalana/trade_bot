import os
import pickle
import logging
import hashlib
from functools import wraps
import inspect


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
    def __init__(self, version_name, optional_attrs=[]):
        self.folder = os.path.join('/home/ytee3/caches', version_name)
        os.makedirs(self.folder, exist_ok=True)
        self.optional_attrs = optional_attrs

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
    
    def __call__(self, func):
        sig = inspect.signature(func)          # capture once, outside wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Construct unique key from func name + args + kwargs
            bound_self = args[0] 
            key_elements = {
                'func': func.__qualname__}
            
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            force_reset = False
            # add every non-self argument in deterministic name=value form
            for name, value in bound.arguments.items():
                if name != "self":
                    key_elements[name]=value
                if name == 'force_reset':
                    force_reset = value

            for k in self.optional_attrs:
                v = getattr(bound_self, k, '')
                key_elements[k] = v

            try:
                # Best-effort serialization of the cache key
                key_str = repr(key_elements)
            except Exception as e:
                logging.warning(f"[CACHE2] Failed to hash args: {e}")
                return func(*args, **kwargs)

            filename = self._file_path(key_str)

            # Check cache hit
            if os.path.exists(filename) and not force_reset:
                try:
                    with open(filename, 'rb') as f:
                        logging.debug(f"[CACHE2] Cache hit: {filename}")
                        return pickle.load(f)
                except Exception as e:
                    logging.warning(f"[CACHE2] Failed to load cache: {e}")

            # Compute and save result
            result = func(*args, **kwargs)
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(result, f)
                    logging.debug(f"[CACHE2] Cache saved: {filename}")
            except Exception as e:
                logging.warning(f"[CACHE2] Failed to save cache: {e}")
            return result
        return wrapper