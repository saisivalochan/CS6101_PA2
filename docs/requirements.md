# Compulsory requirement checklist

Source: ProblemStatement.pdf, seven pages. Checkboxes reflect measured evidence,
not whether code to perform the measurement exists.

## Data (pages 1–2)
- [x] Full SIFT1M base/query/supplied exact truth; L2; exact distances agree (ID ties documented).
- [x] Full Quora and NQ corpus; test query splits; Cohere single-vector embeddings.
- [x] Separate corpus-only PCA per IR dataset; 128 components; saved sample/seed.
- [x] Same fitted transform for corpus and queries; float32 then L2 normalization.
- [x] Exact top-10 saved after projection; BEIR qrels never used for ANN fidelity.
- [x] Summary: corpus/query counts, original/search dimension, N*128*4 raw bytes.
- [x] PCA implementation, sample size, seed, retained explained variance reported.

## Part 1 (page 2)
- [x] IndexFlatL2 and IndexHNSWFlat on every full dataset.
- [x] At least five efSearch operating points; M and efConstruction recorded.
- [x] Each point: both recalls, mean latency, QPS, build time, serialized index bytes.
- [x] Three HNSW 10-recall@10 versus latency plots.
- [x] CPU, thread count, individual/batch mode, warmup and timing subset disclosed.

## Part 2 (pages 2–4)
- [x] Original GreedySearch, RobustPrune, random graph, centroid-nearest entry.
- [x] Complete visited set, bounded candidates, distinct distance/expansion counters.
- [x] Two passes: alpha=1 then configured alpha, shuffled order, reciprocal pruning.
- [x] Reference R=64, L_build=100, alpha=1.2; no parameter deviation.
- [x] Construction measurements near 100K, 250K, 500K before full NQ build.
- [x] All full datasets: parameters, build seconds, degree mean/max, graph/index bytes,
      peak build RSS, entry procedure, seed, representation, language/optimizations.
- [x] Every vertex checked: degree <= R; valid unique non-self neighbors.
- [x] SIFT100K one-query exact/Vamana top-10 IDs, recall, expansions, distances.

## Part 3 (page 5)
- [x] SIFT100K reference, changed-R, changed-alpha graphs, five L_search values each.
- [x] Every study point: parameters, both recalls, latency, QPS, distances, expansions.
- [x] Recall versus L_search and versus latency; effects of R, alpha, L_search discussed.
- [x] One main full Vamana graph per dataset, persisted and reused for search sweeps.
- [x] Three combined Vamana/HNSW latency plots, exact FAISS reference point.
- [x] Comparable-recall table: both recalls, latency, QPS, build time, index size;
      Vamana distance/expansion means. Discuss measured tradeoffs.

## Submission (page 7)
- [ ] One report mirroring parts; all findings and required LLM-use documentation.
- [ ] Actual shareable links to ALL LLM chats used; no invented or placeholder links.
- [ ] Implementation/experiment scripts and report in [Roll1_Roll2].zip.
- [ ] No datasets, embedding collections, credentials, environments or large indices.
- [ ] Roll numbers confirmed; one team member submits through course portal.

## Implementation choices (not assignment requirements)
Seed 42; PCA randomized with min(100000,N) corpus sample; HNSW M=32,
efConstruction=200, efSearch=[10,20,50,100,200]. Vamana reference=(64,100,1.2),
variants=(32,100,1.2),(64,100,1.0); L_search=[10,20,50,100,200].
C++17 with ctypes binding; Python data/FAISS/report orchestration. Query timing
uses one thread, individual k=10 calls, 1000-query seeded subset, three repeats.
No numerical quality or runtime claim is accepted until measured.

Current qualification: all 48 operating points and 12 required builds pass the
final checksum-verifying experiment audit. The isolated NQ rebuild completed,
matched the original graph, and replaced the overlapping construction timing;
original evidence is retained. The test suite passes without skips. The draft report
contains the required experimental sections and eight plots. Final submission
checkboxes remain open because roll numbers, real shareable chat links and the
user's confirmation of complete LLM disclosure are absent.
