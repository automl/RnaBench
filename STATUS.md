# RnaBench — Status (2026-05-30)

## What works

| Component | Status | Notes |
|-----------|--------|-------|
| `import RnaBench` | ✅ | Fixed (see Changes) |
| `RnaFoldingBenchmark` (intra_family) | ✅ | Verified |
| **RNAformer** dim64 | ✅ | f1=0.769, mcc=0.776, wl=0.847 (max_len=80 subset) |
| **UFold** ufold_train.pt | ✅ | f1=0.720, mcc=0.726, wl=0.801 (max_len=80 subset) |
| Folding visualization plots | ✅ | `plots/` populated |
| `RnaDesignBenchmark` inverse_rna_folding | ✅ | Verified |
| **DeterministicGCA** + RNAFold | ✅ | f1=0.756, solved=0.491 (max_len=80 subset) |
| **DeterministicGCA** + ContraFold | ✅ | f1=0.684 |
| **DeterministicGCA** + IpKnot | ✅ | f1=0.743 |
| **DeterministicGCA** + PKiss | ✅ | |
| **DeterministicGCA** + LinearFold-V/C | ✅ | |
| **DeterministicGCA** + RNAStructure Fold | ✅ | Requires `DATAPATH` export |
| **DeterministicGCA** + SPOT-RNA | ✅ | Slow (~27 min for 80-nt subset, TF graph reload) |
| **DeterministicGCA** + MxFold2 | ✅ | Requires install from GitHub source (PyPI wheel broken) |
| Inverse folding visualization plots | ✅ | |
| `examples/f1score_wl_example.py` | ✅ | |
| `examples/visualize_data_length_dist.py` | ✅ | Rewrote to avoid OOM |
| `examples/visualize_representations.py` | ✅ | |
| Conda env `RnaBench` | ✅ | See setup recipe below |

## What doesn't work / not yet verified

| Component | Status | Reason |
|-----------|--------|--------|
| **ProbTransformer** | ✅ | Checkpoints auto-downloaded on first run |
| **RNAformer** large checkpoints (dim512 etc.) | ❓ | Require GPU and bf16; untested without |
| Full-dataset `reproduce_all_working.sh` | ⏳ | Needs 6-12 h CPU or GPU; not run |
| `inter_family` / `biophysical_model` folding tasks | ⏳ | Not run (only `intra_family` tested) |
| `conda env create -f environment.yml` | ❌ | Solver conflict (Python 3.9 + forgi); use recipe below |

## Conda env setup recipe

The original `environment.yml` does not solve. Use this instead:

```bash
conda create -n RnaBench python=3.10 -y
conda activate RnaBench
conda install -c bioconda -c conda-forge viennarna infernal cython -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' 'numpy<2' 'pytorch-lightning<2.1'
pip install forgi==2.2.3 grakel pandas scikit-learn biopython tqdm plotly kaleido
pip install tensorflow  # for SPOT-RNA
pip install 'mxfold2 @ https://github.com/mxfold/mxfold2/archive/refs/tags/v0.1.2.tar.gz'
pip install -e .
```

Key constraints:
- `numpy<2` — GraKeL wheels compiled for NumPy 1.x crash under NumPy 2
- `pytorch-lightning<2.1` — ≥2.1 requires torch ≥2.1
- `setuptools<70` — PL imports `pkg_resources`, removed in setuptools ≥70
- MxFold2 PyPI wheel is broken (missing C extension); install from GitHub source

## External algorithms (one-time setup)

```bash
mkdir -p external_algorithms
# LinearFold
git clone https://github.com/LinearFold/LinearFold external_algorithms/LinearFold && make -C external_algorithms/LinearFold
# IpKnot
gdown 1Oh3kNYbnv_22i4IIYXavPcOo1xSdm1vB -O external_algorithms/ipknot && chmod +x external_algorithms/ipknot
# SPOT-RNA
git clone https://github.com/jaswindersingh2/SPOT-RNA external_algorithms/SPOT-RNA
# VARNA
mkdir -p external_algorithms/VARNA
# download VARNAv3-93.jar to external_algorithms/VARNA/
# the varna shell wrapper is already in external_algorithms/VARNA/varna
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
