from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path
    data_dir: Path
    results_dir: Path

def default_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, data_dir=data_dir, results_dir=results_dir)
