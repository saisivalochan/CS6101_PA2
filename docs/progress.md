# Progress

## Initial environment
Only ProblemStatement.pdf was present. Local Python 3.14 and g++ are available;
pip, NumPy, FAISS, and gcloud are absent. DNS lookup of pypi.org fails in this
sandbox. No cloud project or authenticated access has been provided yet.

## Design adjustment
Use a C ABI plus Python ctypes instead of pybind11. This retains a compiled C++17
core, permits dependency-free local validation, and avoids Python header/package
requirements. It does not change the algorithms or benchmark policy.

## Implemented
- Part A: context documents, reproducible environment, one-VM cloud launch/run/fetch
  scripts with runtime/cost controls, and a source-only upload bundler.
- Part B: SIFT parsing and safe archive extraction; streamed IR embeddings; pinned
  source revision; corpus-only randomized PCA; shared transform, float32 and
  normalization; manifests/checksums; seeded SIFT development/scaling subsets.
- Part C: exact FAISS truth persistence and SIFT verification, HNSW construction,
  consistent warmup/single-thread/individual-query timing, full-query set recall.
- Part D: original C++ GreedySearch, RobustPrune, two-pass build, degree validation,
  centroid-nearest entry, distance caches/tags, serialization and resume checkpoints.
- Parts E/F: fresh-process experiments, 100K/250K/500K scaling gate, three development
  configurations, five search points each, full comparisons and required counters.
- Part G: required plot/table/PDF generation, full LLM transcript appendix, final
  completeness audit and submission archive allowlist. Final generation refuses
  incomplete experimental evidence or missing disclosure details.

## Validation actually executed
- `python3 scripts/build_core.py`: optimized native build succeeded without warnings.
- `python3 -m unittest discover -s tests -v`: 16 tests discovered, 13 passed,
  3 numeric/report integration tests skipped because dependencies are absent.
- `python3 scripts/build_core.py --sanitize` followed by the available unit suite
  with `UBSAN_OPTIONS=halt_on_error=1`: passed; no undefined-behavior diagnostics.
  Restored the normal optimized binary afterward.
- Python syntax compilation and `bash -n` on all cloud scripts passed.
- `python3 -m pa2.cli core-smoke`: 1000 synthetic integer-valued float32 points,
  128 dimensions, 20 queries. At L_search=N, both recalls were 1.0 versus independent
  brute force. See results/local_core_validation.json for measured observations.
  This is validation only, not SIFT/Quora/NQ assignment evidence.
- `python3 -m pa2.cli audit --quick`: correctly reports missing real experiment
  evidence and zero compulsory operating points.
- Full `run` fails clearly on missing dependencies before creating experiment results.

## Remaining work / external dependencies
No real dataset download, full FAISS/Vamana experiment, cloud resource creation,
final PDF, or final submission archive has been completed. Local shell DNS is
blocked and pip/numerical dependencies/gcloud are absent. An authenticated cloud
session and project ID have not been supplied. Use the README Cloud Shell path;
do not paste keys/passwords. No cloud spend has been incurred by these tools.

After cloud access: run bootstrap (which executes all 3 integration tests), then the
full pipeline; inspect scaling before full builds; address real runtime failures;
generate/inspect the draft. Obtain roll numbers, full chat transcripts and genuine
shareable links, generate final report, run final audit, package and retrieve.
Keep datasets/indices on the VM until that final audit and packaging finish.
Recheck the remaining US$50 allowance before any restart. Retrieve outputs and
verify local copies before deleting assignment-created cloud resources.

## Local execution retry — 2026-10-15

Requested execution on this local machine was attempted again, rather than
assuming the earlier environment limitations still applied. No source changes
were needed or made.

- `python3 scripts/build_core.py`: rebuilt `build/libvamana.so` successfully.
- `python3 -m unittest discover -s tests -v`: 16 discovered, 13 passed,
  3 integration tests skipped because numerical/report dependencies are absent.
- `python3 -m pa2.cli --help`: succeeded.
- `python3 -m pa2.cli core-smoke`: succeeded on 1000 synthetic points in 128
  dimensions and 20 queries. Both recalls were 1.0 at L_search=1000. Updated
  `results/local_core_validation.json` and its log. These are validation only.
- `python3 -m pa2.cli run --config configs/main.json`: exit 1 before any data
  preparation or assignment experiment; NumPy, FAISS, sklearn, datasets,
  matplotlib, reportlab and threadpoolctl are missing.
- `python3 -m pa2.cli audit --config configs/main.json`: exit 2, complete=false,
  0 available builds, 0 operating points and 70 missing/invalid evidence entries.
  Updated `results/audit.json`.

Fresh checks found Python 3.14.4 with neither pip nor ensurepip. DNS resolution
failed for pypi.org, huggingface.co and ftp.irisa.fr; a direct curl request to
PyPI also failed with exit 6. No prepared dataset manifests exist here, and no
usable package cache/environment was found in the inspected standard locations.
The machine reports 7.5 GiB RAM and 2 GiB swap; full-run feasibility has not been
measured. No paid operations were started.

Machine/dependency/network evidence is saved in
`results/local_execution_status.json`; test, help, full-run and audit output is
saved in `results/logs/local_*_20261015.log`. Local completion requires network
access to install dependencies and retrieve the full datasets, or a compatible
offline environment with all required datasets. Final report/package generation
also still requires actual roll numbers, full LLM transcripts and genuine
shareable chat links. No final report or submission archive was produced.

## Network-enabled local execution — 2026-10-15

The restarted CLI session can reach PyPI and Hugging Face. A project-local
Python 3.12.15 interpreter and seeded virtual environment were installed using
a checksum-verified uv wheel from PyPI. Package installation is in progress;
the previous missing-dependency status above describes the earlier attempt.
The mixed-case Windows path is treated as read-only by the sandbox configured
with a lowercase workspace path, so local installation/download commands needed
approved execution outside that sandbox. No paid cloud resources were used.

The original SIFT archive was downloaded from the pinned HTTPS mirror
`zjd2000/ANN_SIFT` revision `8956d7e088e0b79b6901c234632e801fa37d731d`
because the canonical FTP endpoint was unavailable from this network. SHA-256
`92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`
was verified. The 168,280,445-byte archive contains the original 516,000,000-byte
base file, 5,160,000-byte query file, and 4,040,000-byte supplied ground-truth
file. See `data/raw/sift/download.json` for the URL, checksum and member sizes.
This is a download milestone, not a completed data-preparation or experiment.

Read all seven pages of the assignment using pypdf, confirming the compulsory
scope and the page-7 full-transcript/shareable-link requirement. Submission
identity and genuine chat links/transcripts were requested while execution
continues; they must not be invented or omitted from the final report.

Local dependency installation completed using
`.venv/bin/python -m pip install --cache-dir /tmp/cs6101-pip-cache -e . pypdf`.
Exact versions are saved in `results/environment.lock.txt`. The native core was
rebuilt with `python3 scripts/build_core.py`.

The first complete numerical test run found a missing SIFT100K trace when a
caller's parameter dictionary also contained `L_search`. The reference-graph
comparison in `pa2/experiments.py` now compares only R, L_build and alpha. The
existing integration test covers this failure. After the fix,
`.venv/bin/python -m unittest discover -s tests -v` ran all 16 tests in 25.595 s:
all passed, with no skips. Logs retain both the original failure and the passing
rerun in `results/logs/local_full_tests*_20261015.log`. These temporary synthetic
integration fixtures are not assignment experiment evidence.

Started the full local command:
`.venv/bin/python -m pa2.cli run --config configs/main.json --sift-archive data/raw/sift/sift.tar.gz`.
It must still complete the full-data preparation, scaling gate, all builds and
48 required operating points before experimental completion can be claimed.

Full SIFT1M preparation completed: `data/sift1m/manifest.json` records N=1,000,000,
Q=10,000, D=128, no normalization, and checksums for the float32 vectors, ID map,
and supplied top-10 truth. `001_prepare_sift1m__.log` records 26.501 s wall time.
The seeded SIFT100K subset is prepared and its reference graph build is running.

`python3 scripts/export_llm_chats.py` exported visible messages and tool activity
from three local Codex sessions with this workspace as their recorded directory.
The snapshots and hashes are in `llm_transcripts/`; internal reasoning and
system/developer instructions are excluded. The active session must be refreshed
after work finishes. Other-product chats, actual share URLs, and confirmation of
complete disclosure still require the user's input.

The first real graph finished: SIFT100K reference R=64, L_build=100, alpha=1.2,
seed 42; construction 294.217737912 s, peak build RSS 127,479,808 bytes,
average out-degree 56.96636, maximum 64, graph 26,006,459 bytes. The completed
graph was validated for every vertex and saved with its SHA-256 and provenance
in `results/builds/sift100k__vamana_R64_B100_a1.2.json`. The 250K scaling build
has started. No query operating points have been measured yet.

Before any real query timing, Vamana evaluation was adjusted to materialize its
entire corpus as a resident contiguous float32 array, like FAISS's loaded vector
storage. This excludes first-touch mapped-file disk reads from the in-memory
latency comparison. The timing policy and report now disclose this. Syntax
checks passed; the integration suite must be rerun at a point when no graph
build is running. This Python evaluation change does not alter construction or
invalidate the completed graph.

`submission_details.draft.json` maps the three exported local chats to their
actual transcript paths. It deliberately has no roll numbers or invented share
URLs, and disclosure completeness is false. It is not final submission details.

The 250K scaling graph completed in 997.970487487 s, peak build RSS 246,038,528
bytes, maximum out-degree 64, with all vertex/edge checks passing. See
`results/builds/sift250000__vamana_R64_B100_a1.2.json`. The 500K graph is in its
second pass; the empirical scaling estimate is not complete yet. No query timing
is running alongside these builds.

All required scaling builds have now completed. The 500K graph took
1987.553048760 s, with peak build RSS 443,293,696 bytes, mean degree 58.11205,
maximum degree 64, and a 130,006,474-byte serialized graph. All vertex/edge checks
passed. Its metadata is `results/builds/sift500000__vamana_R64_B100_a1.2.json`.
The pipeline has moved to full Quora preparation; no corpus has been reduced.
The integration suite is being rerun during data preparation, after construction
has ended, to validate the resident-corpus timing change.

## Overnight local results — 2026-10-16

While the test-rerun approval was pending, the original full pipeline continued.
Both IR datasets are fully prepared from revision
`f018922a46a51388348a307ec2fef28019a33026`: Quora N=522,931/Q=10,000 and
NQ N=2,681,468/Q=3,452. Each PCA used 100,000 uniformly sampled corpus rows,
seed 42, sklearn 1.9.1 randomized PCA; retained explained variance is
0.6142749190 (Quora) and 0.5955088139 (NQ). The scaling gate passed, estimating
21,212.52 s total main construction against 80,343.48 s remaining allowance.

At 03:00 UTC, 43/48 operating points were saved: all exact/HNSW baselines,
all 15 SIFT100K ablation points, and all SIFT1M/Quora Vamana points. Full Vamana
construction took 4090.251899358 s for SIFT1M and 1402.009116561 s for Quora.
The NQ graph was in its second pass. SIFT100K query row 0 matched all exact top-10
IDs at L_search=100, with 102 expansions and 2863 distinct distance evaluations.
These are real dataset results, separate from all temporary integration fixtures.

The delayed test approval caused the rerun to start at approximately 03:00 UTC
during NQ construction, rather than during data preparation as intended. All
16 tests passed in 55.009 s, including the resident-corpus evaluation change;
see `results/logs/local_full_tests_final_20261016.log`. No required assignment
query timing overlapped it, but the NQ construction measurement did. This was
an execution error and must not be hidden or corrected by subtracting time.

`scripts/finalize_local.py` waits for the original pipeline, preserves its NQ
index/build/query evidence, obtains an isolated NQ rebuild with identical inputs,
and verifies identical serialized content after normalizing only the elapsed-time
header. It then promotes the clean index/measurement, repeats the five NQ query
points sequentially, runs all tests after timing, audits all evidence, and
generates the report. Missing genuine disclosure details still block final
packaging. Original measurements remain under `results/build_validation/original`
and `indices/build_validation/original`; no result is silently discarded.
Two new tests check graph-equivalence normalization and conflicting-evidence
preservation. They are scheduled with the final post-measurement test suite.

Direct validation of the saved full IR artifacts also passed. The saved PCA
sample IDs exactly match the seeded uniform corpus choices; recomputed means
from all 100,000 selected raw corpus rows match the saved means exactly for both
datasets. Applying each saved mean/components to raw query embeddings and then
normalizing reproduces prepared queries within 1.35e-7 absolute error. Sampled
corpus unit-norm errors are at most 1.20e-7; PCA components are orthonormal within
1e-4. See `results/data_validation.json` for actual counts and observations.

The two new finalization safety tests were run separately without query
benchmarking and passed (2 tests, 0.024 s). The report now explicitly explains
the measured SIFT ground-truth tie differences, compares R/alpha variants at a
fixed L_search, embeds a Unicode-capable font, and uses a logarithmic latency
axis for combined plots so the exact reference does not hide the ANN curves.
These report changes will be checked by the final full integration suite.

The original full pipeline exited 0 at approximately 04:29 UTC on 2026-10-16.
Its checksum-verifying audit passed: 48 operating points, 12 required builds,
four prepared datasets including SIFT100K, and no missing/invalid evidence. The
initial NQ build took 12247.571371140 s, mean degree 61.08748566, maximum 64,
graph bytes 697,188,134, and peak RSS 2,161,512,448 bytes. Its short integration
overlap remains documented above, and these measurements are being preserved.

The isolated NQ validation build started at 04:30 UTC. No query benchmarks or
integration timing are running alongside it. Report wording also explicitly
states that construction elapsed times can include first-touch reads of mapped
corpus pages; dataset preparation/opening and final serialization are excluded.
Query comparisons materialize the full corpus before timing, as already noted.

## Local experimental completion — 2026-10-16T08:23:08.141671+00:00

The full experiment audit passes with 48 required operating points. All full corpora were used. A clean isolated NQ rebuild replaced the initial overlapping construction measurement, with graph equivalence verified and initial evidence preserved. All tests passed after construction and query measurement finished.

Report: `results/report_draft.pdf`. Submission complete: False.
Remaining submission requirements: Submission details file is absent (roll numbers, LLM documentation, chat links).

## Final artifact inspection — 2026-10-16

- Finalizer exited 0. Clean NQ construction took 13,510.288884716 s; normalized
  serialized graph SHA-256 is
  `8dce5c55b5ccf537cb9827d6f4f9288520bfd0d240d7f150124c49d6f35e9781`.
  Initial construction/index and all five initial query/timing records remain
  under the documented `build_validation/original` directories. The clean build
  and all five replacement NQ query measurements ran sequentially.
- `.venv/bin/python -m unittest discover -s tests -v` passed all 18 tests in
  48.227 s, without skips, after construction and query measurement finished.
  Full log: `results/build_validation/logs/final_tests.log`.
- The checksum-verifying final audit reports complete=true, 48 operating points,
  12 required builds, and no missing/invalid experimental evidence.
- Inspected `results/metrics.csv`: 48 real measurement rows. Inspected the draft
  PDF with pypdf and rendered all 14 main-content pages with PyMuPDF installed
  only under `/tmp` for preview. Dataset/PCA metadata, protocol, baseline tables,
  all eight plots, construction/scaling/trace, ablations and full comparisons
  are present and legible. The remaining pages preserve exported LLM transcripts.
- Updated README and experimental requirement checkboxes. The draft explicitly
  distinguishes local transcript snapshots from complete final disclosure.
  No paid operations were started. No final submission archive is claimed:
  roll numbers, genuine links for all chats and complete-disclosure confirmation
  still require the user. Complete `submission_details.draft.json`, include other
  used chats, and save it as `submission_details.json` before final generation.

The final transcript refresh exposed a 1.9-million-character inline screenshot
encoding in a visible tool result. PDF line wrapping became prohibitively slow,
so that regeneration was interrupted; the earlier valid draft remained intact.
The exporter now decodes inline images into checksum-named attachments without
losing their bytes or the surrounding text. Report appendices display those
attachments, final packaging includes them, and final-report auditing verifies
their hashes. Two focused regression tests passed in 0.004 s, covering byte/text
preservation and rejection of changed attachments. The full finalizer is rerun
with the completed clean-build marker, so it reuses the validated index and cached
query points before testing all 20 tests and regenerating the readable draft.

All 20 tests passed on that rerun. Report generation then encountered the short
truncated image-data excerpt printed by the earlier diagnostic command, in
addition to the complete screenshot. It is not a valid displayable PNG. The
report now labels any undecodable image and preserves its original bytes and
hash, while displaying complete image attachments normally. No text or original
attachment bytes are removed to make report generation succeed.

## Local experimental completion — 2026-10-16T09:38:34.931854+00:00

The full experiment audit passes with 48 required operating points. All full corpora were used. A clean isolated NQ rebuild replaced the initial overlapping construction measurement, with graph equivalence verified and initial evidence preserved. All tests passed after construction and query measurement finished.

Report: `results/report_draft.pdf`. Submission complete: False.
Remaining submission requirements: Submission details file is absent (roll numbers, LLM documentation, chat links).
