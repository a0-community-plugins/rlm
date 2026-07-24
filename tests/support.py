from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(
    os.getenv("A0_TEST_PROJECT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
