# Result-figure scripts

The plotters stay with their experiments and use the supplied measurements.

| Figure | Source |
| --- | --- |
| 2: capacity and SIMD width | [plot_fig2.py](../experiments/capacity_motivation/plot_fig2.py) |
| 7: GEMM evaluation | [plot_fig7.py](../experiments/vl_length/plot_fig7.py) |
| 8: alternative mechanisms | [make_fig_ladder_live.py](../experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py) |
| 9: layer evaluation | [plot_fig9.py](../experiments/end2end/scripts/plot_fig9.py), [renderer](../experiments/end2end/scripts/render_fig9.py) |
| Table IV | [plot_table4.py](../experiments/end2end/scripts/plot_table4.py) |

From the repository root, run `bash reproduce.sh` as described in the
[main README](../README.md). Outputs go to `results/`.

Architecture-diagram sources for Figures 3--6 are maintained in the
[paper repository](https://github.com/JingqunZhang/CacheFlex-Direct-Software-Managed-Access-to-Higher-Level-Cache-for-Scalable-Vector-Support-2-/tree/main/figures).
