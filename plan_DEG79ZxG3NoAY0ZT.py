import lamindb as ln
import bionty as bt
import pandas as pd
import anndata as ad
from pathlib import Path

ln.connect("ishitajain9717/mutation-registry")
ln.track()

# Ingest the raw count matrix
path = ln.core.datasets.file_tsv_rnaseq_nfcore_salmon_merged_gene_counts(
    populate_registries=True
)
ln.Artifact(path, description="Merged Bulk RNA counts").save()

# Load and reshape to tidy AnnData
artifact = ln.Artifact.get(description="Merged Bulk RNA counts")
df = artifact.load()

df = df.T
var = pd.DataFrame({"gene_name": df.loc["gene_name"].values}, index=df.loc["gene_id"])
adata = ad.AnnData(df.iloc[2:].astype("float32"), var=var)

# Define a schema and validate
bt.settings.organism = "saccharomyces cerevisiae"
bulk_schema = ln.Schema(itype=bt.Gene.stable_id, otype="AnnData").save()
curator = ln.curators.AnnDataCurator(adata, bulk_schema)
curator.validate()

# Save the curated artifact
curated_af = curator.save_artifact(description="Curated bulk RNA counts")

# Attach assay and organism labels
efs = bt.ExperimentalFactor.lookup()
organism = bt.Organism.lookup()
features = ln.Feature.lookup()
curated_af.labels.add(efs.rna_seq, features.assay)
curated_af.labels.add(organism.saccharomyces_cerevisiae, features.organism)

curated_af.describe()

ln.finish()
