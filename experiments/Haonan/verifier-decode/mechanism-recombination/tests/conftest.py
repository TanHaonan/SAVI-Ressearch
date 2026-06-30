"""Pytest path setup for the mechanism-recombination test suite.

Adds the verifier-decode parent (so ``decode_core`` imports) and the
``mechanism-recombination`` dir itself (so ``domains`` / ``gen_fair`` / ``arms_ext`` /
``run_mechanism`` import) to ``sys.path``. The domain bundles handle isolating the
frozen M1 / M2 ``core`` packages internally.
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_MECH_DIR = os.path.dirname(_TESTS_DIR)              # mechanism-recombination/
_PACKAGE_PARENT = os.path.dirname(_MECH_DIR)         # verifier-decode/

for _p in (_PACKAGE_PARENT, _MECH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
