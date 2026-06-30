"""pytest bootstrap for correlated-coverage tests.

Put the experiment dir (the parent of this ``tests/`` package) on ``sys.path`` so
``import core_boot`` / ``import mock_backend`` resolve, then importing ``core_boot``
self-bootstraps the countdown-decode substrate (``import core``) per the integration
brief's recipe.
"""

import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

import core_boot  # noqa: E402,F401  (places countdown-decode on sys.path; smoke import)
