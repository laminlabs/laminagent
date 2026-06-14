import shutil
from pathlib import Path

import lamindb as ln
import pytest
from testutils import TESTDB1_DEV_DIR, TESTDB1_NAME, TESTDB1_STORAGE


@pytest.fixture(scope="session", autouse=True)
def setup_testdb1():
    dbroot = Path(TESTDB1_STORAGE)
    if dbroot.exists():
        print("removing existing testdb1 storage location and database")
        shutil.rmtree(dbroot)
        ln.setup.delete(dbroot, force=True)
    dev_dir = Path(TESTDB1_DEV_DIR)
    if dev_dir.exists():
        print("removing existing testdb1 development directory")
        shutil.rmtree(dev_dir)
    dev_dir.mkdir(parents=True, exist_ok=True)
    ln.setup.init(name=TESTDB1_NAME, storage=dbroot, modules="bionty")
    ln.setup.settings.dev_dir = dev_dir
