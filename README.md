# CacheFlex Artifact

This repository contains the CacheFlex gem5 model, benchmark sources,
experiment scripts, and measurements used for Figures 2, 7, 8, and 9 and
Table 4 of the paper.

## Generate the paper figures

Python 3.11 or 3.12 and the packages in `requirements.txt` are required:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source .venv/bin/activate
bash reproduce.sh
```

The figures, their JSON data, and Table 4 are written to `results/`. Plotting
the supplied measurements does not require building gem5 or the benchmark
binaries.

To build gem5 and rerun the simulations, follow [SETUP.md](SETUP.md), then run:

```bash
bash run_all.sh -j 8
```

## Contents

- `gem5/`: CacheFlex simulator implementation
- `spm_tools/`: CacheFlex instruction encoding support
- `kernels/`: benchmark sources
- `experiments/`: experiment configurations, measurements, and plotters
- [figures/](figures/README.md): index of result-figure scripts
- `reproduce.sh`: generates Figures 2, 7, 8, and 9 and Table 4

Architecture-diagram sources are maintained in the
[paper repository](https://github.com/JingqunZhang/CacheFlex-Direct-Software-Managed-Access-to-Higher-Level-Cache-for-Scalable-Vector-Support-2-/tree/main/figures).

## License

CacheFlex's own contributions are released under the MIT License; see
[LICENSE](LICENSE). The SVE FP16 GEMM microkernels are derived from the Arm
Compute Library and retain its MIT notice. The bundled gem5 simulator and its
third-party components remain under their original licenses; see
`gem5/LICENSE` and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
