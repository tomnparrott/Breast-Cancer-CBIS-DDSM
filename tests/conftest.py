import sys
from pathlib import Path

# Put the repo root on sys.path so pytest can import project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))