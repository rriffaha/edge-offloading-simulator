"""Allow tests to import the simulator's existing flat Python modules."""
import sys
from pathlib import Path

SIMULATOR = Path(__file__).resolve().parents[1] / "calibrated_model"
sys.path.insert(0, str(SIMULATOR))
