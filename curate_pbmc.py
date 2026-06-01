
import lamindb as ln
import bionty as bt
import anndata as ad

ln.connect("ishitajain9717/mutation-registry")

adata = ln.core.datasets.anndata_pbmc68k_reduced()

# Standardize gene index from symbols to Ensembl IDs
adata.var["ensembl_gene_id"] = bt.Gene.standardize(
    adata.var.index,
    field=bt.Gene.symbol,
    return_field=bt.Gene.ensembl_gene_id,
    organism="human",
)

# Set Ensembl ID as the new index
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index().set_index("ensembl_gene_id")

# Map cell type labels to canonical ontology terms
cell_types = bt.CellType.public().lookup()

name_mapping = {
    "Dendritic cells": cell_types.dendritic_cell.name,
    "CD19+ B": cell_types.b_cell_cd19_positive.name,
    "CD4+/CD45RO+ Memory": cell_types.cd4_positive_alpha_beta_memory_t_cell.name,
    "CD8+ Cytotoxic T": cell_types.cytotoxic_t_cell.name,
    "CD4+/CD25 T Reg": cell_types.regulatory_t_cell.name,
    "CD14+ Monocytes": cell_types.cd14_positive_monocyte.name,
    "CD56+ NK": cell_types.natural_killer_cell.name,
    "CD8+/CD45RA+ Naive Cytotoxic": cell_types.naive_t_cell.name,
    "CD34+": cell_types.hematopoietic_stem_cell.name,
}

adata.obs["cell_type"] = adata.obs["cell_type"].map(name_mapping)

# Check for NaN values after mapping
assert adata.obs["cell_type"].isna().sum() == 0, "Some cell types could not be mapped"

# Define the cell_type feature if it does not exist
if not ln.Feature.filter(name="cell_type").exists():
    ln.Feature(name="cell_type", dtype=bt.CellType).save()

# Save the validated artifact
try:
    ln.track()
    artifact = ln.Artifact.from_anndata(
        adata,
        key="scrna/pbmc68k_reduced.h5ad",
        description="Curated scRNA dataset with validated cell types and Ensembl gene ids",
        schema="ensembl_gene_ids_and_valid_features_in_obs",
    ).save()

    artifact.describe()
    ln.finish()
except SystemExit as e:
    print("Validation error:", e)
