# B: Data
PDF pages 1–2. Inputs: SIFT1M archive; CohereLabs/beir-embed-english-v3 Quora/NQ
corpus train splits and queries test splits. Outputs: float32 C-contiguous .npy
vectors, original-ID JSONL mappings, PCA .npz, exact shape/source/checksum manifest.
Use corpus-only uniform PCA sample min(100000,N), seed 42, randomized PCA 128.
Save selected row IDs, mean, components, explained variance and sklearn version.
Transform corpus and queries using identical fitted mean/components; cast then
normalize; fail on zero norm/nonfinite embeddings. SIFT remains raw float32 L2.
SIFT100K is a uniform subset with saved parent row mapping, not supplied SIFT1M
ground truth. Do not invent query counts or corpus rows from approximate sizes.
Tests: dimensions, corpus-only sample, normalization, mapping and truth provenance.
