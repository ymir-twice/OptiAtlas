"""Bounded, ordered concurrency for independent dataset validation reads."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice


def validate_workers(workers, *, name="validation_workers"):
    if type(workers) is not int or workers < 1:
        raise ValueError(f"{name} must be a positive integer")


def ordered_bounded_map(function, values, *, workers):
    """Yield in input order, keeping at most twice ``workers`` futures alive.

    Unlike Executor.map on older Python versions, this does not eagerly submit
    an entire inventory. Errors are observed in input order, as in serial mode.
    """
    validate_workers(workers)
    if workers == 1:
        yield from map(function, values)
        return
    values = iter(values)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gold-validation") as executor:
        pending = deque()
        try:
            pending.extend(executor.submit(function, value) for value in islice(values, 2 * workers))
            while pending:
                result = pending.popleft().result()
                yield result
                for value in islice(values, 1):
                    pending.append(executor.submit(function, value))
        finally:
            for future in pending:
                future.cancel()
