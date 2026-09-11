# Figure sources

The table links each figure to its editable source. Diagram PDFs are kept
beside their sources; result-chart scripts remain with their experiments.

| Figure | Source |
| --- | --- |
| 2: capacity and SIMD width | [plot_fig2.py](../experiments/capacity_motivation/plot_fig2.py) |
| 3: bridge datapaths | [fig3.tex](diagrams/fig3.tex), [PDF](diagrams/fig3.pdf) |
| 4: architecture overview | [fig4.drawio](diagrams/fig4.drawio), [PDF](diagrams/fig4.pdf) |
| 5: dual-LSQ ordering | [draw_fig5_lsq.py](diagrams/draw_fig5_lsq.py), [PDF](diagrams/fig5.pdf) |
| 6: in-cache transactions | [draw_fig6_transactions.py](diagrams/draw_fig6_transactions.py), [PDF](diagrams/fig6.pdf) |
| 7: GEMM evaluation | [plot_fig7.py](../experiments/vl_length/plot_fig7.py) |
| 8: alternative mechanisms | [make_fig_ladder_live.py](../experiments/software_alternatives/twosize_ladder/make_fig_ladder_live.py) |
| 9: layer evaluation | [plot_fig9.py](../experiments/end2end/scripts/plot_fig9.py), [renderer](../experiments/end2end/scripts/render_fig9.py) |
| Table IV | [plot_table4.py](../experiments/end2end/scripts/plot_table4.py) |

## Generate result charts

From the repository root, run `bash reproduce.sh` as described in the
[main README](../README.md). Outputs go to `results/`.

## Edit or regenerate diagrams

Figure 3 uses pdfLaTeX with TikZ/PGF, geometry, xcolor, and PSNFSS fonts.
Figures 5 and 6 use Python and Matplotlib; Figure 5 also needs PyMuPDF.
From the repository root:

```bash
python3 -m pip install -r figures/requirements.txt
mkdir -p results/diagrams
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=results/diagrams figures/diagrams/fig3.tex
python3 figures/diagrams/draw_fig5_lsq.py results/diagrams/fig5.pdf
python3 figures/diagrams/draw_fig6_transactions.py results/diagrams/fig6.pdf
```

The Python scripts also write PNG previews. Open `diagrams/fig4.drawio`
in diagrams.net to edit Figure 4 and export it as a cropped PDF. The supplied
diagram PDFs are the versions used in the paper; editor exports can vary
slightly with fonts and renderer versions.
