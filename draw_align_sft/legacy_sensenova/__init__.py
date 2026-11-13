"""Differentiable training support for the native SenseNova U models.

The vendored inference implementation is loaded explicitly by path; this package
does not import or change the evaluation repository at module import time.
"""

def run(cfg, logger=None):
    from .runner import run as _run
    return _run(cfg, logger=logger)


def preflight(cfg):
    from .runner import preflight as _preflight
    return _preflight(cfg)

__all__ = ["preflight", "run"]
