import os
import shutil
from pathlib import Path

import lamindb as ln
import pytest
from laminagent.setup import setup as laminagent_setup
from testutils import TESTDB1_DEV_DIR, TESTDB1_NAME, TESTDB1_STORAGE


@pytest.fixture(scope="session", autouse=True)
def setup_testdb1():
    dev_dir = Path(TESTDB1_DEV_DIR)
    if dev_dir.exists():
        print("removing existing testdb1 development directory")
        shutil.rmtree(dev_dir)
    dev_dir.mkdir(parents=True, exist_ok=True)

    # for metric tracking, we're logging runs against main
    # to the lamindata instance
    if (
        os.getenv("GITHUB_ACTIONS") == "true"
        and os.getenv("GITHUB_EVENT_NAME") == "push"
        and os.getenv("GITHUB_REF") == "refs/heads/main"
    ):
        ln.setup.settings.dev_dir = dev_dir
        laminagent_setup(verbose=False)
        return

    storage_root = Path(TESTDB1_STORAGE)
    if storage_root.exists():
        print("removing existing testdb1 storage location and database")
        shutil.rmtree(storage_root)
        ln.setup.delete(TESTDB1_NAME, force=True)
    ln.setup.init(name=TESTDB1_NAME, storage=storage_root, modules="bionty")
    ln.setup.settings.dev_dir = dev_dir
    laminagent_setup(verbose=False)
