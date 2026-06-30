"""decode_core — domain-agnostic trellis decoder shared across SAVI domains.

This package generalizes the M1 Countdown trellis decoder so ANY domain plugs in via
the small ``Domain`` interface (see ``decode_core.domain``). The decoder
(``decode_core.decode``) is identical in behavior to M1's ``core/decode.py`` but takes
a ``domain`` object + an injected ``sample`` callable instead of importing the
Countdown modules directly. It imports NO domain code and NO m1 code: a domain is
supplied by the caller, the sampling backend is baked into the ``sample`` closure.

Behavior-preservation against M1 is proved by the parity test in
``decode_core/tests/test_countdown_parity.py``.
"""

from . import domain, decode

__all__ = ["domain", "decode"]
