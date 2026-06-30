"""Shared evaluator core for both SAVI lines.

Domain-agnostic metrics over behavioral KEYS (strings) or distributions.
The domain map Phi is applied by the CALLER before calling anything here; this
package never imports a domain.

Canonical source. Each line vendors a copy into its own core/_deps/metrics/ for
self-contained reproduction (see vendor_metrics.py). Edit here, then re-vendor;
never hand-edit a vendored copy.
"""
from . import diversity, coverage, markov, calibration, passk

__all__ = ["diversity", "coverage", "markov", "calibration", "passk"]
