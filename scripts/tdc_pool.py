"""Bounded process scheduling and aggregate process-tree memory measurement."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import threading
import time

import psutil


class TreeBudget:
    def __init__(self, c):
        self.c, self.start = c, time.perf_counter()
        self.proc = psutil.Process()
        self.peak, self.maximum_processes = 0, 1
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def sample(self):
        processes = [self.proc] + self.proc.children(recursive=True)
        rss = 0
        alive = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
                alive += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.peak = max(self.peak, rss)
        self.maximum_processes = max(self.maximum_processes, alive)

    def monitor(self):
        while not self.stop.wait(.05):
            self.sample()

    def check(self):
        self.sample()
        if self.peak > self.c['max_rss_bytes'] or time.perf_counter()-self.start > self.c['max_seconds']:
            raise RuntimeError('Registered aggregate process-tree budget exceeded')

    def close(self):
        self.stop.set()
        self.thread.join()
        self.sample()
        return dict(seconds=time.perf_counter()-self.start, peak_sampled_process_tree_rss_bytes=self.peak,
                    maximum_observed_processes=self.maximum_processes, sample_seconds=.05,
                    interpretation='Sum of live RSS, includes launcher/children and shared pages; not an OS exact aggregate peak')


def pooled(function, jobs, workers, c, label):
    tree = TreeBudget(c)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(function, job): job for job in jobs}
            while pending:
                tree.check()
                ready, _ = wait(pending, timeout=.5, return_when=FIRST_COMPLETED)
                for future in ready:
                    job = pending.pop(future)
                    result = future.result()
                    results.append(result)
                    print(f'{label}: {len(results)}/{len(jobs)} completed - {result["identity"]}', flush=True)
        return results, tree.close()
    finally:
        if not tree.stop.is_set():
            tree.close()
