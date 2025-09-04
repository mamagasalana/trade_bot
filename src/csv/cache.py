import os
import pickle
import logging
import hashlib
from functools import wraps
import inspect
import time
import  zlib
from peewee import SqliteDatabase, Model, DatabaseProxy, TextField, BlobField, OperationalError, FloatField, SQL
from queue import Queue, Empty, Full
from threading import Thread, Event
import random
import json

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
    def __init__(self, version_name, optional_attrs=[], levels=3, chars=2):
        """
        Args:
            version_name: name of cache folder
            optional_attrs: attrs to include in key
            levels: number of subdirectory levels (default 3)
            chars: characters per level (default 2, good for hex)
        """
        self.folder = os.path.join('/home/ytee3/caches', version_name)
        os.makedirs(self.folder, exist_ok=True)
        self.optional_attrs = optional_attrs
        self.levels = levels
        self.chars = chars

    def _hash_key(self, key):
        """Convert the key into a safe hashed filename."""
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    # def _file_path(self, key):
    #     """Get the full file path for a given key."""
    #     return os.path.join(self.folder, self._hash_key(key))
    def _file_path(self, key):
        """Shard cache files into multiple subdirectories."""
        hashed = self._hash_key(key)
        parts = [hashed[i:i+self.chars] for i in range(0, self.levels*self.chars, self.chars)]
        path = os.path.join(self.folder, *parts)
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, hashed)

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
                if name == 'force_reset':
                    force_reset = value
                elif name != "self":
                    key_elements[name]=value

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
                try:
                    time.sleep(1)
                    with open(filename, 'wb') as f:
                        pickle.dump(result, f)
                        logging.debug(f"[CACHE2] Cache saved: {filename}")
                except Exception as e:
                    logging.warning(f"[CACHE2] Failed to save cache: {e}")
            return result
        return wrapper


db_proxy = DatabaseProxy()

class CacheRow(Model):
    k = TextField(primary_key=True)
    v = BlobField()
    meta = TextField(null=True)         # JSON dump of key_elements
    updated_at = FloatField(index=True) # optional but useful
    class Meta:
        database = db_proxy


class CACHE3:
    def __init__(self, version_name, optional_attrs=None, compress=False, compress_level=3):
        self.optional_attrs = optional_attrs or []
        self.compress = bool(compress)
        self.compress_level = int(compress_level)

        base_dir = "/home/ytee3/caches"
        os.makedirs(base_dir, exist_ok=True)
        fname = os.path.join(base_dir, f"{version_name}.db")

        self.db = SqliteDatabase(fname, pragmas={
            "journal_mode": "wal",
            "synchronous": "normal",
            "busy_timeout": 60000,
            "temp_store": "memory",
        })

        db_proxy.initialize(self.db)
        self.db.connect(reuse_if_open=True)
        self.db.create_tables([CacheRow])


        queue_maxsize=10000
        writer_batch_size=256
        writer_max_delay=0.02      # seconds to wait to grow a batch
        enqueue_blocking=False     # False: don't block when full
        enqueue_timeout=0.0        # seconds (only if blocking)
        
        self._async = True
        self._q = Queue(maxsize=queue_maxsize)
        self._stop = Event()
        self._writer_batch_size = int(writer_batch_size)
        self._writer_max_delay = float(writer_max_delay)
        self._enqueue_blocking = bool(enqueue_blocking)
        self._enqueue_timeout = float(enqueue_timeout)

        if self._async:
            self._writer = Thread(target=self._writer_loop,
                                  name=f"CACHE3-writer-{version_name}",
                                  daemon=True)
            self._writer.start()
        else:
            self._writer = None

    def __len__(self):
        with self.db.connection_context():
            return CacheRow.select().count()

    def _writer_loop(self):
        """
        Drain queue and upsert rows in small batches using insert_many.
        Coalesce duplicate keys within a batch (last write wins).
        """
        with self.db.connection_context():
            while not self._stop.is_set():
                try:
                    first = self._q.get(timeout=0.1)
                except Empty:
                    continue

                # Coalesce by key: h -> (blob, meta_json)
                batch = {}
                n_polled = 0

                h, v, meta = first
                batch[h] = (v, meta); n_polled += 1

                t0 = time.time()
                while (len(batch) < self._writer_batch_size and
                    (time.time() - t0) < self._writer_max_delay):
                    try:
                        h, v, meta = self._q.get_nowait()
                        batch[h] = (v, meta)   # last write wins
                        n_polled += 1
                    except Empty:
                        break

                now = time.time()
                rows = [{'k': h, 'v': v, 'meta': m, 'updated_at': now} for h, (v, m) in batch.items()]

                # Single-statement upsert for the whole batch
                # SQLite 3.24+ supports ON CONFLICT ... DO UPDATE with "excluded".
                try:
                    for attempt in range(6):
                        try:
                            with self.db.atomic():
                                (CacheRow
                                .insert_many(rows, fields=[CacheRow.k, CacheRow.v, CacheRow.meta, CacheRow.updated_at])
                                .on_conflict(
                                    conflict_target=[CacheRow.k],
                                    update={
                                        CacheRow.v: SQL('excluded.v'),
                                        CacheRow.meta: SQL('excluded.meta'),
                                        CacheRow.updated_at: SQL('excluded.updated_at'),
                                    })
                                .execute())
                            break  # success
                        except OperationalError as e:
                            msg = str(e).lower()
                            if "locked" in msg or "busy" in msg:
                                time.sleep(min(0.02 * (2 ** attempt) + random.random() * 0.005, 0.5))
                                continue
                            logging.warning("[CACHE3] writer batch failed: %s", e)
                            break
                except Exception as e:
                    logging.warning("[CACHE3] writer batch failed: %s", e)
                finally:
                    for _ in range(n_polled):
                        self._q.task_done()
                        
    def flush(self, timeout=None):
        """Block until all queued writes have been committed."""
        start = time.time()
        self._q.join()
        if timeout is not None and (time.time() - start) > timeout:
            logging.debug("[CACHE3] flush timed out")


    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _deserialize(self, blob: bytes):
        if not blob:
            return None
        try:
            if blob.startswith(b"ZC3\0"):
                blob = zlib.decompress(blob[4:])
            return pickle.loads(blob)
        except Exception as e:
            logging.debug(f"[CACHE3] deserialize failed: {e}")
            return None

    def _serialize(self, obj) -> bytes:
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.compress:
            try:
                b = zlib.compress(b, self.compress_level)
                b = b"ZC3\0" + b
            except Exception:
                pass
        return b

    def __setitem__(self, key, value, meta_dict=None):
        h = self._hash_key(str(key))
        blob = self._serialize(value)
        meta_json = json.dumps(meta_dict or {}, default=str)
        try:
            self._q.put((h, blob, meta_json), block=self._enqueue_blocking, timeout=self._enqueue_timeout)
        except Full:
            # queue saturated — pick your policy:
            # 1) best-effort drop (fastest):
            logging.warning("[CACHE3] queue full; dropping write for key=%s", h)



    def __getitem__(self, key):
        h = self._hash_key(str(key))
        with self.db.connection_context():
            row = CacheRow.get_or_none(CacheRow.k == h)
        if row is None:
            logging.debug(f"[CACHE3] miss {h}")
            return None
        try:
            return self._deserialize(row.v)
        except Exception as e:
            logging.debug(f"[CACHE3] deserialize error for {h}: {e}")
            return None

    def __contains__(self, key):
        h = self._hash_key(str(key))
        with self.db.connection_context():
            return CacheRow.select().where(CacheRow.k == h).exists()

    def __call__(self, func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            bound_self = args[0] if args else None
            key_elements = {'func': func.__qualname__}

            force_reset = False
            for name, value in bound.arguments.items():
                if name == 'force_reset':
                    force_reset = value
                elif name != 'self':
                    key_elements[name] = value

            if bound_self is not None:
                for attr in self.optional_attrs:
                    key_elements[attr] = getattr(bound_self, attr, '')

            try:
                key_str = repr(key_elements)   # RAW key (not hashed here)
            except Exception as e:
                logging.warning(f"[CACHE3] Failed to repr key: {e}")
                return func(*args, **kwargs)

            if not force_reset and key_str in self:
                return self[key_str]

            result = func(*args, **kwargs)
            self.__setitem__(key_str, result, meta_dict=key_elements)
            return result

        return wrapper

