import lamindb as ln
import anndata as ad
import bionty as bt
import sys

# Connect to the registry
ln.connect("ishitajain9717/mutation-registry")

# Load the dataset
adata = ln.core.datasets.anndata_pbmc68k_reduced()

# Track the run
ln.track()

# Attempt validation immediately
try:
    ln.Artifact.from_anndata(
        adata,
        key="scrna/dataset.h5ad",
        schema="ensembl_gene_ids_and_valid_features_in_obs",
    ).save()
except SystemExit as e:
    print("Validation error:", e)

# Inspect the var index
print(adata.var.index[:10])
print(adata.var.index.name)

# Standardize gene index from symbols to Ensembl IDs if necessary
if adata.var.index.name != "ensembl_gene_id":
    adata.var["ensembl_gene_id"] = bt.Gene.standardize(
        adata.var.index,
        field=bt.Gene.symbol,
        return_field=bt.Gene.ensembl_gene_id,
        organism="human",
    )
    adata.var.index.name = "symbol"
    adata.var = adata.var.reset_index().set_index("ensembl_gene_id")

# Inspect cell type labels
print(adata.obs["cell_type"].unique())

# Map cell type labels to canonical ontology terms
name_mapping = {}
for label in adata.obs["cell_type"].unique():
    try:
        cell_type = bt.CellType.public().lookup(label)
        name_mapping[label] = cell_type.name
    except KeyError:
        print(f"Label '{label}' could not be mapped to a canonical cell type.")
        # Stop and report the unmapped label
        raise ValueError(f"Label '{label}' could not be mapped to a canonical cell type.")

adata.obs["cell_type"] = adata.obs["cell_type"].map(name_mapping)

# Define the cell_type feature if it does not exist
if not ln.Feature.filter(name="cell_type").exists():
    ln.Feature(name="cell_type", dtype=bt.CellType).save()

# Save the validated artifact
try:
    artifact = ln.Artifact.from_anndata(
        adata,
        key="scrna/dataset.h5ad",
        description="Curated scRNA dataset with validated cell types and Ensembl gene ids",
        schema="ensembl_gene_ids_and_valid_features_in_obs",
    ).save()
    artifact.describe()
    ln.finish()
except SystemExit as e:
    print("Validation error:", e)
