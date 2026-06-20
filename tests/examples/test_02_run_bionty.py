import bionty as bt


def test_bionty_basic() -> None:
    """Sanity-check: can bionty look up 'B cell' from the Cell Ontology source?"""
    record = bt.CellType.from_source(name="B cell")
    assert record is not None
