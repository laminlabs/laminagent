import shutil
import sys
from pathlib import Path

import lamindb as ln
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def setup_lamindb():
    ln.setup.init(storage="./testagentdb")
    yield
    shutil.rmtree("./testagentdb")
    ln.setup.delete("testagentdb", force=True)
