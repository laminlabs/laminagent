import lamindb as ln
ln.track()
import bionty as bt
import anndata as ad
import pandas as pd

# Connect to the registry
ln.connect("ishitajain9717/mutation-registry")

# Load the dataset and populate registries
rnaseq_data = ln.core.datasets.file_tsv_rnaseq_nfcore_salmon_merged_gene_counts(populate_registries=True)

# Load the data into a pandas dataframe
rnaseq_df = pd.read_csv(rnaseq_data, sep='\t')

# Reshape to AnnData tidy format
adata = ad.AnnData(X=rnaseq_df.values, var=pd.DataFrame(index=rnaseq_df.columns), obs=pd.DataFrame(index=rnaseq_df.index))

# Validate against the schema and save the artifact
try:
    artifact = ln.Artifact.from_anndata(
        adata,
        schema="saccharomyces_cerevisiae_gene_counts",
    )
    artifact.save()
    print("Artifact saved successfully.")
except SystemExit as e:
    print("Validation error:", e)

ln.finish()
