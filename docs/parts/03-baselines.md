# C: Exact and HNSW
PDF pages 1–2,5. Inputs: prepared dataset manifests. Outputs: exact IDs/distances,
serialized exact/HNSW indices, CSV metrics, query timing repetitions and metadata.
L2 for all. SIFT1M supplied truth is authoritative; verify FAISS agreement and
record tie evidence rather than replacing truth. SIFT100K/IR require new exact
truth in their own spaces. k=10 calls, 1-recall@1 from rank 1; top-10 set overlap/10.
HNSW M32/efConstruction200, efSearch10,20,50,100,200. One thread, individual calls,
warmup, identical saved timing subset min(1000,Q), three repeats; median run chosen.
Recall uses full Q. Build/load times are not query latency. Peak RSS is measured in
fresh subprocesses. Explicitly distinguish graph bytes from searchable index bytes.
Test with synthetic exact neighbors and known recall sets; reloading must agree.
