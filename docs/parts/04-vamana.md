# D: Vamana
PDF pages 2–4 and linked DiskANN paper. Own C++ implementation only. Python ctypes
adapter holds the input array alive. Inputs: float32 vectors, R/L_build/alpha/seed.
Graph: fixed-width uint32 rows + degrees; no self/duplicate/out-of-range edges.
GreedySearch starts with entry, repeatedly expands closest unexpanded candidate,
unions neighbors, retains closest L, and returns k plus ALL expanded vertices.
Distance cache and expansion stamps live for the whole query, including evictions.
RobustPrune merges existing edges and visited candidates, excludes p, nearest first,
removes candidates when alpha^2*d2(selected,candidate)<=d2(p,candidate), stops at R.
Entry: nearest point to double-precision corpus mean. Two shuffled passes: 1,alpha.
Snapshot outgoing neighbors for reciprocal insertion, pruning on overflow.
Reference R64/Lbuild100/alpha1.2, seed42. Construction stays sequential; parallelize
independent preprocessing/experiments only. Save graph with provenance and periodic
resumable build checkpoints. Query L_search is independent and >=k.
Validate against literal Python reference; counters, eviction, ties, pruning alpha,
degree bounds, reciprocal edges, serialization and checkpoint/resume.
