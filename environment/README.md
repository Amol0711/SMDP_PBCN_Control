# Environments

The requirements describe the recorded installed numerical versions, not universal platform compatibility or a newly completed installation. Core study paths use `requirements-core.txt`; biological paths additionally use `requirements-biological.txt`. The recorded PyTorch distribution was the CPU build 2.10.0+cpu; an appropriate lawful platform wheel source is required. No package files, wheel caches or font binaries are bundled.

A separate TeX installation with pdfLaTeX, standalone, TikZ and PGFPlots is needed for native figures; the guide uses its listed standard LaTeX packages. Shell escape remains disabled. The standard-library transport/help/output-copy operations need no NumPy or Torch.

The exact historical biological-continuation descriptor is retained at `pd_smac/analysis/config/recorded_environment.json`. It records Python 3.13.5, AMD EPYC 9V74, Linux 6.18.44, NumPy 2.3.5, the recorded Torch build, thread/determinism settings and further identity fields. The code compares the complete descriptor, not just the CPU model or package versions. Do not edit that descriptor to make a different host pass.

Where a numerical command is authorized, record effective thread settings rather than assuming environment variables were honored. Typical single-thread settings are `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1`. These settings do not override environment eligibility or grant execution authorization.

## Observed validation environment, 22 October 2026

The installed validation used CPython 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Numba 0.65.1, llvmlite 0.47.0, psutil 7.2.2, threadpoolctl 3.6.0, jsonschema 4.26.0, mpmath 1.3.0, pandas 2.2.3 and PyTorch 2.10.0+cpu. NumPy/SciPy BLAS pools and PyTorch intra/inter-operation thread counts were checked at one; deterministic PyTorch execution was recorded. These are observed reproduction settings, not recommendations to update dependencies.

The actual platform was Linux 6.18.55 with glibc 2.41 and the AMD EPYC 9V74 processor. The unchanged historical continuation guard requires Linux 6.18.44 and rejected the mismatch. Matching a CPU or package version does not make a mismatching full descriptor eligible.

An isolated virtual environment with system and user numerical packages excluded was actually created and probed. The core dependency installation failed at package-index DNS resolution, leaving NumPy unavailable; a subsequent isolated witness failed for that missing dependency. This is not demonstrated dependency incompatibility, and no clean numerical installation pass is recorded. See the validation report for the exact scope and unchanged-payload binding.
