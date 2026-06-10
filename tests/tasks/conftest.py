import shutil
from pathlib import Path

import lamindb as ln
import pytest


@pytest.fixture(scope="session")
def setup_lamindb():
    ln.setup.init(storage="./testagentdb")
    yield
    shutil.rmtree("./testagentdb")
    ln.setup.delete("testagentdb", force=True)


@pytest.fixture(scope="session", autouse=True)
def setup_testdb1():
    dbroot_str = "./testdb1"
    if Path(dbroot_str).exists():
        shutil.rmtree(dbroot_str)
        ln.setup.delete(dbroot_str, force=True)
    runs_root = Path("./testdb1-runs")
    if runs_root.exists():
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    ln.setup.init(storage=dbroot_str, modules="bionty")
