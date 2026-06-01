import lamindb as ln
ln.track()
import bionty as bt

ln.connect("ishitajain9717/mutation-registry")

# Load bulk RNA-seq dataset
rna_seq_data = ln.core.datasets.file_tsv_rnaseq_nfcore_salmon_merged_gene_counts(populate_registries=True)

# Reshape to AnnData tidy format
import anndata as ad
rna_seq_adata = ad.AnnData(rna_seq_data)

# Validate against schema with itype=bt.Gene.stable_id for Saccharomyces cerevisiae
schema_uid = ln.core.schemas.get_schema_uid(schema_type='rnaseq', itype=bt.Gene.stable_id, organism='saccharomyces_cerevisiae')
try:
    artifact = ln.Artifact.from_anndata(rna_seq_adata, key='rna_seq_dataset', schema=schema_uid).save()
    print('Validation passed, artifact saved.')
except SystemExit as e:
    print('Validation error:', e)

# Attach RNA-seq assay and Saccharomyces cerevisiae organism labels
rna_seq_adata.obs['assay'] = 'rna_seq'
rna_seq_adata.obs['organism'] = 'saccharomyces_cerevisiae'

ln.finish()
