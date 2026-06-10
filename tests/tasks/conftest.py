import shutil

import lamindb as ln
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_testdb1():
    ln.setup.init(storage="./testdb1", modules="bionty")
    yield
    shutil.rmtree("./testdb1")
    ln.setup.delete("testdb1", force=True)
