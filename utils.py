import gc
import gzip
import shutil
from collections import defaultdict
from functools import partial
from multiprocessing import Process, Event, Value
import os
import time
from tempfile import NamedTemporaryFile, TemporaryFile
from timeit import default_timer

import feather
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from pyarrow.lib import ArrowInvalid


class Timer:
    """Simple util to measure execution time.
    Examples
    --------
    >>> import time
    >>> with Timer() as timer:
    ...     time.sleep(1)
    >>> print(timer)
    00:00:01
    """
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = default_timer() - self.start

    def __str__(self):
        return self.verbose()

    def __float__(self):
        return self.elapsed

    def verbose(self):
        if self.elapsed is None:
            return '<not-measured>'
        return self.format_elapsed_time(self.elapsed)

    @staticmethod
    def format_elapsed_time(value: float):
        return time.strftime('%H:%M:%S', time.gmtime(value))
    
    
class MemoryTrackingProcess(Process):
    """A process that periodically measures the amount of RAM consumed by another process.
    
    This process is stopped as soon as the event is set.
    """
    def __init__(self, pid, event, **kwargs):
        super().__init__()
        self.p = psutil.Process(pid)
        self.event = event
        self.max_mem = Value('f', 0.0)
        
    def run(self):
        mem_usage = []
        while not self.event.is_set():
            info = self.p.memory_info()
            mem_bytes = info.rss
            mem_usage.append(mem_bytes)
            time.sleep(0.05)
        self.max_mem.value = np.max(mem_usage)


class MemoryTracker:
    """A context manager that runs MemoryTrackingProcess in background and collects 
    the information about used memory when the context is exited.
    """
    def __init__(self, pid=None):
        pid = pid or os.getpid()
        self.start_mem = psutil.Process(pid).memory_info().rss
        self.event = Event()
        self.p = MemoryTrackingProcess(pid, self.event)
    
    @property
    def memory(self):
        return self.p.max_mem.value - self.start_mem
    
    def __enter__(self):
        self.p.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.event.set()
        self.p.join()

        
class GC:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, ext_tb):
        self.collected = gc.collect()
        

class VisualStyle:
    def __init__(self, config, default=None):
        if default is None:
            default = plt.rcParams
        self.default = default.copy()
        self.config = config
        
    def replace(self):
        plt.rcParams = self.config
    
    def override(self, extra=None):
        plt.rcParams.update(self.config)
        if extra is not None:
            plt.rcParams.update(extra)

    def restore(self):
        plt.rcParams = self.default


class NotebookStyle(VisualStyle):
    def __init__(self):
        super().__init__({
            'figure.figsize': (11, 8),
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'font.size': 16
        })


def generate_dataset(n_rows, num_count, cat_count, max_nan=0.1, max_cat_size=100):
    """Randomly generate datasets with numerical and categorical features.

    The numerical features are taken from the normal distribution X ~ N(0, 1).
    The categorical features are generated as random uuid4 strings with
    cardinality C where 2 <= C <= max_cat_size.

    Also, a max_nan proportion of both numerical and categorical features is replaces
    with NaN values.
    """
    dataset, types = {}, {}

    def generate_categories():
        from uuid import uuid4
        category_size = np.random.randint(2, max_cat_size)
        return [str(uuid4()) for _ in range(category_size)]

    for col in range(num_count):
        name = f'n{col}'
        values = np.random.normal(0, 1, n_rows)
        nan_cnt = np.random.randint(1, int(max_nan * n_rows))
        index = np.random.choice(n_rows, nan_cnt, replace=False)
        values[index] = np.nan
        dataset[name] = values
        types[name] = 'float32'

    for col in range(cat_count):
        name = f'c{col}'
        cats = generate_categories()
        values = np.array(np.random.choice(cats, n_rows, replace=True), dtype=object)
        nan_cnt = np.random.randint(1, int(max_nan * n_rows))
        index = np.random.choice(n_rows, nan_cnt, replace=False)
        values[index] = np.nan
        dataset[name] = values
        types[name] = 'object'

    return pd.DataFrame(dataset), types


def size_of(filename, unit=1024**2):
    return round(os.stat(filename).st_size / unit, 2)


def save_feather(df, fname, compress=False, compresslevel=7):
    if compress is False:
        df.to_feather(fname)
    else:
        tmpf = NamedTemporaryFile()
        df.to_feather(tmpf.name)
        with open(tmpf.name, mode='rb') as f_in:
            with gzip.open(fname, compresslevel=compresslevel, mode='wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        tmpf.close()


def read_feather(source, columns=None, use_threads=True):
    try:
        return feather.read_dataframe(source, columns, use_threads)
    except ArrowInvalid:
        fin = open(source, 'rb')
        tmpf = TemporaryFile('w+b')
        with gzip.GzipFile(fileobj=fin, mode='rb') as fgz:
            data = fgz.read()
            tmpf.write(data)
        tmpf.seek(0)
        return feather.read_dataframe(tmpf, columns, use_threads)


def get_save_load(df, fmt):
    if fmt == 'feather':
        return partial(save_feather, df), read_feather
    return getattr(df, f'to_{fmt}'), getattr(pd, f'read_{fmt}')


def benchmark(list_of_formats, data_size=1_000_000, n_num=15, n_cat=15, n_rounds=20,
              as_category=False):
    """Runs dataset saving/loading benchamrk using formts from the list_of_formats.

    Each round a new random dataset is generated with data_size observations.
    The measurements for each of the rounds are concatenated together and returned
    as a single data frame.

    Parameters:
        list_of_formats: A list of tuples in the format (<format_name>, [<params_dict>]).
            The <format_name> should be one of the pandas supported formats.
        data_size: A number of samples in the generated dataset.
        n_num: A number of numerical columns in the generated dataset.
        n_cat: A number of categorical columns in the generated dataset.
        n_rounds: A number of randomly generated datasets to test the formats.
        as_category: If True, then categorical columns will be converted into
            pandas.Category type before saving.

    """
    runs = []

    for i in range(n_rounds):
        print(f'Benchmarking round #{i + 1:d}')
        print('\tgenerating dataset...')
        dataset, _ = generate_dataset(data_size, n_num, n_cat)

        if as_category:
            print('\tconverting categorical columns into pandas.Category')
            cat_cols = dataset.select_dtypes(include=object).columns
            dataset[cat_cols] = dataset[cat_cols].fillna('none').astype('category')

        benchmark = []

        for case in list_of_formats:
            fmt, params = case if len(case) == 2 else (case[0], {})

            with GC():
                print('\ttesting format:', fmt)
                filename = f'random.{fmt}'
                save, load = get_save_load(dataset, fmt)
                results = defaultdict(int)
                results['format'] = fmt
                results['filename'] = filename

                with MemoryTracker() as tracker:
                    with Timer() as timer:
                        save(filename, **params)
                results['size_mb'] = size_of(filename)
                results['save_ram_delta_mb'] = tracker.memory / (1024 ** 2)
                results['save_time'] = float(timer)

                with MemoryTracker() as tracker:
                    with Timer() as timer:
                        _ = load(filename)
                results['load_ram_delta_mb'] = tracker.memory / (1024 ** 2)
                results['load_time'] = float(timer)

                benchmark.append(results)

            run = pd.DataFrame(benchmark)
            run['run_no'] = i
            runs.append(run)

    benchmark = pd.concat(runs, axis=0)
    benchmark.reset_index(inplace=True, drop=True)
    return benchmark