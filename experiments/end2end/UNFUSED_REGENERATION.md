# Unfused-attention reference cells: regeneration provenance (2026-07)

## Why

An audit found that the eight unfused-attention reference cells
(`llama_T{256,4096}_{unfused,spm_unfused}` in both VL blocks) had been
generated with a truncated command line (`-o 'T 32 8 64 1'` — blocking
parameters defaulted, no attention-mode token) that did not match the
arguments declared in `best.json`
(`'T 32 8 64 1 128 256 128 0 noncausal'`). A fresh run with the declared
arguments could therefore never reproduce those cells, and four of them are
the baseline denominators of the Figure 9 speedups.

## What was done

All eight cells were re-simulated with the DECLARED best.json arguments and
the shipped binaries (same gem5, same frozen machine configuration —
config.ini diff-verified against the neighbouring cells). Logs carry full
argv and `ATTENTION_MODE=noncausal` receipts; both arms of each pair
produce identical checksums, as required for the exact (unfused) algorithm:

| cell pair | BL / CF cold (ms) | shared checksum |
|---|---|---|
| vl4  T256  | 3.24 / 3.25   | 83.622936606 |
| vl16 T256  | 1.71 / 1.70   | 83.622936606 |
| vl4  T4096 | 1105.17 / 1100.86 | 183.041253626 |
| vl16 T4096 | 803.23 / 815.61   | 183.041199982 |

`reference_integrity.py` and `plot_fig9_real.py` were updated to validate
these cells on the standard (non-legacy) path; manifests were re-sealed.

## Effect on rendered outputs (erratum)

The regenerated cells run ~25% faster than the originals (modern binaries,
correct blocking arguments), so Figure 9's BL / CF (unfused-attention)
bars shift by that amount in absolute time. The BL:CF unfused pairs remain
at parity (ratios 0.98-1.00) exactly as before, and the flash-attention
cells, Table 4, and Figures 2/7/8 are byte-identical to the previous
rendering. Because the regenerated cells are the Figure 9 baseline
denominators, the speedup-vs-BL annotations of the FA configurations
decrease correspondingly (e.g. LLaMA T4096/VL16 CF+FA moves from the
~1.8x class to 1.62x as rendered); the capacity-pressure conclusions are
unaffected. The original cells are preserved outside the artifact for
auditability.
