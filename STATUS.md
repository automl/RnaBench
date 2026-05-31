# RnaBench — Status (2026-05-30)

## What doesn't work / not yet verified

| Component | Status | Reason |
|-----------|--------|--------|
| **ProbTransformer** | ✅ | Checkpoints auto-downloaded on first run |
| **RNAformer** large checkpoints (dim512 etc.) | ❓ | Require GPU and bf16; untested without |
| Full-dataset `reproduce_all_working.sh` | ⏳ | Needs 6-12 h CPU or GPU; not run |
| `inter_family` / `biophysical_model` folding tasks | ⏳ | Not run (only `intra_family` tested) |
| `conda env create -f environment.yml` | ❌ | Solver conflict (Python 3.9 + forgi); use recipe below |

## Conda env setup recipe

```bash
conda env create -f environment.yml
conda activate RnaBench
./install_external_algorithms.sh
python -m RnaBench.download
./reproduce_all_fast.sh # or ./reproduce_all_working.sh if you have the time/resources (GPU recommended)
```

## How to reproduce

```bash
# Quick smoke test (~15 min, max_len=80):
./reproduce_all_fast.sh

# Full pipeline (hours):
./reproduce_all_working.sh
```

Both scripts set `PATH`, `MPLBACKEND=Agg`, and `DATAPATH` correctly.

## Changes made and why

| File | Change | Why |
|------|--------|-----|
| `RnaBench/benchmarks.py` | Moved `torchvision` import inside `get_iterators()` | torchvision not always installed; top-level import broke every benchmark import |
| `RnaBench/benchmarks.py` | Fixed result filename (used `folding_algo_name` instead of object) | Object stringified as `<obj at 0x...>`, making filenames unreadable |
| `RnaBench/download.py` | Moved `import gdown` inside the function | gdown not always installed |
| `RnaBench/lib/datasets.py` | Moved `import dask` inside `.dask()` / `.to_pandas()` | dask not always installed; top-level import broke everything |
| `RnaBench/lib/datasets.py` | Fixed duplicate `@property def struc_itos` | Second copy silently returned `struc_stoi` (wrong dict) |
| `RnaBench/lib/feature_extractors.py` | Moved `import forgi` inside function | forgi not always installed |
| `RnaBench/lib/feature_extractors.py` | `np.NaN`/`np.NAN` → `np.nan` | Removed in NumPy 2 |
| `RnaBench/lib/rna_folding_algorithms/DL/ufold/data_generator.py` | `np.bool` → `bool` | Removed in NumPy 2 |
| `RnaBench/lib/data/build_cm.py` | Wrapped body in `main()` + `__main__` guard | Module-level code ran BLAST/Infernal pipeline on any import |
| `RnaBench/lib/rna_folding_algorithms/DL/RNAformer/rnaformer_wrapper.py` | Same `main()` guard | Module-level code downloaded data and ran benchmarks on import |
| `RnaBench/lib/rna_folding_algorithms/DL/RNAformer/rnaformer.py` | `flash_attn = False` unconditionally | flash_attn is CUDA-only and not bundled; checkpoints are bf16 and triggered `FlashAttention2d.unpad_input` (undefined) |
| `RnaBench/lib/rna_folding_algorithms/DL/ProbTransformer/probtransformer.py` | Fixed `os.makedirs` path for checkpoint download | `makedirs("checkpoints")` created the wrong directory; checkpoints then downloaded to path that didn't exist |
| `environment.yml` | Added `pytorch-lightning<2.1`, `setuptools<70`; fixed `mxfold2` to GitHub source | These were missing/broken, causing env creation to produce a non-working env |
| `install_external_algorithms.sh` | Rewrote to be a complete, idempotent one-shot setup script | Old version was incomplete (no VARNA, fragile paths, not idempotent) |
| `RnaBench/lib/visualization.py` | Rewrote `get_detailed_statistics()` to use integer counters | String `+=` in a 615k-row loop → O(n²) RAM, consumed 9 GB and OOMed |
| `examples/visualize_data_length_dist.py` | Read only `length` column, skip full statistics pass | Full statistics pass OOMed on biophysical_model_train (615k rows) |
| `examples/visualize_folding_predictions.py` | Auto-detect results dir | Hardcoded `inter_family`; crashes if you ran `intra_family` |
| `environment.yml` | python 3.9→3.10, removed broken channels/packages, loosed numpy pin | Solver could not resolve on any channel |
| `external_algorithms/VARNA/varna` | New shell wrapper for VARNA jar | VARNA is a Java jar with no CLI entry point on PATH |
| `reproduce_*_working.sh`, `reproduce_all_fast.sh` | New wrapper scripts | Original scripts missing PATH/MPLBACKEND/DATAPATH setup |
