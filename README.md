# CacheFlex Artifact

This artifact provides the code, fixed configurations, reference results, and
workflows for Figure 2, Figure 7, Figure 8, Figure 9, and Table 4 (flash
scaling). Kernel provenance and the result-comparison policy are documented
in [docs/CLAIMS.md](docs/CLAIMS.md).

## Quick check from reference results

The no-simulation check needs only Python 3.11--3.12 and the packages in
`requirements.txt`:

```bash
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else "ERROR: CacheFlex requires Python 3.11 or 3.12")'
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source setup_env.sh
bash scripts/render_reference.sh
```

It does not require the ARM toolchain, QEMU, a gem5 build, the ROI-marker
object, or the smoke tests. The environment command may therefore report
those full-run components as missing; they are not used by this check. On a
typical host, rendering the supplied results takes only a few minutes.

The script verifies all shipped reference manifests and result cells,
including the complete Figure 9 grid (the Table 4 companion pool is
validated by the configuration and value checks in
`plot_table4_flash.py --check`), and renders:

- `fig2_real.pdf`, `fig2_real.png`, and `fig2_data.json`
- `fig7_real.pdf`, `fig7_real.png`, and `fig7_data.json`
- `fig8_real.pdf`, `fig8_real.png`, and `fig8_data.json`
- `fig9_real.pdf`, `fig9_real.png`, and `fig9_data.json`
- `table4_data.json`, `table4_flash.tex`, and `table4_flash.md` (Table 4,
  rendered from the frozen reference and the self-contained companion pool
  in `experiments/end2end/table4/`)
- `OUTPUT_MANIFEST.sha256`, binding every rendered figure to its plot data

All outputs go to `results/generated/reference_outputs/`. The command never
modifies `results/reference/` or any supplied reference figure.

## Run all paper experiments

Complete [SETUP.md](SETUP.md) once for the full simulation path. It installs
the required tools, builds gem5, and checks the environment.

From the repository root, run:

```bash
bash run_all.sh -j 4
```

`-j N` sets the maximum total number of gem5 simulations that may run in
parallel. Use a value from 1 to 16; the default is `-j 1`, and the effective
value is also capped at the host's reported CPU count.

Expect roughly 1–2 days of wall clock for the full run at `-j 12`–`-j 16`
on a recent server: the largest individual cells (vl4 LLaMA T=4096
FFN/attention) each take several hours, and a fully loaded run keeps every
slot busy. Individual cells are capped at 24 h wall clock
(`RH_CELL_TIMEOUT` seconds to override).

The command builds the benchmarks and then runs the paper experiments in
this order:

1. Figure 2: capacity sensitivity
2. Figure 7: SIMD-width sensitivity
3. Figure 8: software alternatives
4. Figure 9: component-composed prefill layers

The four stages run 21, 42, 12, and 108 selected cells, respectively.
The default path runs only the selected configurations in each
`best.json`; it does not perform parameter sweeps.
The frozen SPM issue configuration uses two load slots at VL=4/8 and one
load slot at VL=16, with one store slot at every vector length.

The regenerated paper outputs are written under
`results/generated/paper_outputs/`:

- `fig2_real.pdf`, `fig2_real.png`, and `fig2_data.json`
- `fig7_real.pdf`, `fig7_real.png`, and `fig7_data.json`
- `fig8_real.pdf`, `fig8_real.png`, and `fig8_data.json`
- `fig9_real.pdf`, `fig9_real.png`, and `fig9_data.json`
- `OUTPUT_MANIFEST.sha256`, binding every rendered figure to its plot data

After plotting, `run_all.sh` compares the fresh time, performance, energy,
speedup, and normalized-ratio values with the supplied paper reference.  The
fixed limit is 5% relative error.  Figure 9 metrics are gated one-sidedly:
regressions beyond 5% fail, while improvements of any size pass and are
printed as `[IMPROVED]`, because the shipped kernels include disclosed
post-submission optimizations — see [docs/CLAIMS.md](docs/CLAIMS.md) for
the kernel provenance and comparison policy.  Figure structure, workload
and cell identities must match exactly, and the existing correctness
checks remain strict.  A failed comparison returns a nonzero status and
keeps the fresh outputs for inspection.

To run an experiment's declared tuning grid, add `--sweep` to its `run.sh`.
These optional grids are not part of the default paper workflow. On a
fresh code-only extraction, use the root `run_all.sh` workflow so the
required benchmark binaries are built first.

## License

CacheFlex's own contributions (`kernels/`, `scripts/`, `spm_tools/`,
`experiments/`, and the CacheFlex modifications) are released under the MIT
License; see [`LICENSE`](LICENSE). The SVE FP16 GEMM microkernels in `kernels/`
are derived from the Arm Compute Library (Arm Limited, MIT License); that
upstream notice is retained in the affected source files and in
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES). The bundled gem5 simulator
(`gem5/`) and its third-party components (`gem5/ext/`) remain under their
original licenses; see `gem5/LICENSE` and the respective subdirectories.
