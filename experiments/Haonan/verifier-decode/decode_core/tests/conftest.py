"""Pytest path setup for the decode_core test suite.

Two import surfaces are wired here:

1. The shared package itself. We add ``decode_core``'s PARENT to ``sys.path`` so
   ``from decode_core import ...`` resolves cleanly no matter the invocation cwd
   (mirrors how the M1 tests rely on ``core.*`` being importable from the m1 root).

2. The FROZEN M1 modules, for the test-only Countdown adapter and the parity test.
   These are imported *by the tests only* — ``decode_core`` itself never imports m1.
   The M1 package lays out its modules under ``m1-countdown-trellis/core`` and they
   import each other as ``core.countdown`` / ``core.sampler``; so we add the m1 line
   ROOT (the dir that contains ``core/``) to ``sys.path`` and import ``core.*`` and
   ``core``-rooted ``decode`` via the m1 package. To avoid the m1 line's ``core``
   package shadowing nothing in decode_core (decode_core has no ``core`` package),
   this is safe.
"""

import sys
from pathlib import Path

# decode_core/tests/conftest.py -> decode_core/ -> verifier-decode/
_TESTS_DIR = Path(__file__).resolve().parent
_DECODE_CORE_DIR = _TESTS_DIR.parent
_PACKAGE_PARENT = _DECODE_CORE_DIR.parent           # verifier-decode/  (holds decode_core/)
_M1_ROOT = (_PACKAGE_PARENT / "m1-countdown-trellis").resolve()  # holds core/

for p in (str(_PACKAGE_PARENT), str(_M1_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
