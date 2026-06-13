import shutil
import sys
from pathlib import Path

import lamindb as ln
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def setup_lamindb():
    runs_root = Path("./testdb1-runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    ln.setup.init(storage="./testdb1", modules="bionty")
    yield
    shutil.rmtree("./testdb1", ignore_errors=True)
    try:
        ln.setup.delete("testdb1", force=True)
    except Exception:  # noqa: S110
        pass
