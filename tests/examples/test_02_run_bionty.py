import bionty as bt


def test_bionty_basic() -> None:
    """Sanity-check: can bionty look up 'B cell' from the Cell Ontology source?"""
    df = bt.CellType.public().to_dataframe()
    print(df)
    print(df.loc[df["name"].str.startswith("B cell")])
    assert "B cell" in df["name"].values
    record = bt.CellType.from_source(name="B cell")
    assert record is not None
