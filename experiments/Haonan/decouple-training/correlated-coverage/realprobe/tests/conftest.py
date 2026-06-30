import sys
from pathlib import Path

# Put the realprobe package dir on the path so ``import realprobe_core`` works from
# pytest regardless of the invoking cwd. (realprobe_core itself boots the countdown
# ``core`` package onto sys.path at import time.)
_REALPROBE = Path(__file__).resolve().parent.parent
if str(_REALPROBE) not in sys.path:
    sys.path.insert(0, str(_REALPROBE))
