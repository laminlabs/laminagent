import sys
from pathlib import Path

import lamindb as ln
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INSTANCE_SLUG = "laminlabs/lamindata"


@pytest.fixture(scope="session")
def setup_lamindb():
    # Connect to the real, existing instance. Do NOT init/delete here — this is
    # a live instance and the tracked run is expected to write real records.
    ln.connect(INSTANCE_SLUG)
    yield
