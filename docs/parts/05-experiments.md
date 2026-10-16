# E/F: Experiments
PDF pages 4–5. Scaling 100K/250K/500K SIFT subsets precedes large builds. Extrapolate
from measurements and report evidence before NQ. Never silently downsample full
corpora or change reference params. Three SIFT100K graphs: (64,100,1.2),
(32,100,1.2), (64,100,1.0); five Lsearch10,20,50,100,200 each. One main graph per
full dataset, same five search points. Reuse compatible completed indices/checkpoints.
Inputs: data/truth, validated core. Outputs: build JSON, per-point metrics CSV,
timing repeats, SIFT100K query trace, figures. Record CPU/libraries/threads, peak
RSS, degrees, serialized graph and total searchable bytes. Do not benchmark while
another build is running. Closest-recall pairs must show their actual recall gap.
Validation: artifact fingerprints, expected rows, complete full-query recall,
all required plots, valid counters and degree checks. No fabricated monotonicity
claims: discuss measured behavior even when noisy or nonmonotonic.
