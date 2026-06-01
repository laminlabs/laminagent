import lamindb as ln
import bionty as bt
import anndata as ad

ln.connect('ishitajain9717/mutation-registry')
ln.track()

# Load the dataset
adata = ln.core.datasets.anndata_pbmc68k_reduced()

# Attempt validation right away — read the error carefully
try:
    ln.Artifact.from_anndata(
        adata,
        key='pbmc68k_reduced.h5ad',
        schema='ensembl_gene_ids_and_valid_features_in_obs',
    ).save()
except SystemExit as e:
    print('Validation error:', e)

# Inspect the var index before doing anything
print(adata.var.index[:10])
print(adata.var.index.name)

# If the index looks like gene symbols, standardize it
if adata.var.index.name == 'symbol':
    adata.var['ensembl_gene_id'] = bt.Gene.standardize(
        adata.var.index,
        field=bt.Gene.symbol,
        return_field=bt.Gene.ensembl_gene_id,
        organism='human',
    )
    # Set Ensembl ID as the new index
    adata.var.index.name = 'ensembl_gene_id'
    adata.var = adata.var.reset_index().set_index('ensembl_gene_id')

# Inspect cell type labels
print(adata.obs['cell_type'].unique())

# Map cell type labels to canonical ontology terms
cell_types = bt.CellType.public().lookup()
name_mapping = {}
for label in adata.obs['cell_type'].unique():
    try:
        name_mapping[label] = cell_types[label].name
    except KeyError:
        # If a label has no match, search more carefully
        print(f'No match found for {label}. Searching...')
        search_results = bt.CellType.public().search(label).head(5)
        print(search_results)
        # If after searching you still cannot find a match, stop and report the unmapped labels to the user
        print(f'Could not map {label}. Please add it manually.')
        exit()
adata.obs['cell_type'] = adata.obs['cell_type'].map(name_mapping)

# Define the cell_type feature if it does not exist
if not ln.Feature.filter(name='cell_type').exists():
    ln.Feature(name='cell_type', dtype=bt.CellType).save()

# Save the validated artifact
artifact = ln.Artifact.from_anndata(
    adata,
    key='pbmc68k_reduced.h5ad',
    description='Curated pbmc68k reduced dataset with validated cell types and Ensembl gene ids',
    schema='ensembl_gene_ids_and_valid_features_in_obs',
).save()
artifact.describe()
ln.finish()
