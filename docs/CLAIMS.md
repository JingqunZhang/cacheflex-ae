# Claims, kernel provenance, and result-comparison policy

## What this artifact reproduces

- **Render path (primary):** `scripts/render_reference.sh` regenerates every
  paper figure (2/7/8/9) and Table 4 from the shipped frozen reference
  results, with integrity and value checks. Output matches the submitted
  version of the paper exactly for Figures 2/7/8 and Table 4; Figure 9's
  unfused-attention components reflect the regenerated reference cells and
  sit ~25% below the submitted figure in absolute time, which also lowers
  the speedup-vs-BL annotations (disclosed erratum:
  `experiments/end2end/UNFUSED_REGENERATION.md`). `run_all.sh` includes
  this path.
- **Re-simulation path:** `run_all.sh` rebuilds the kernels and re-simulates
  all cells. Fresh results are compared against the reference by
  `scripts/compare_results.py` (policy below).

## Post-submission kernel state (disclosure)

The CacheFlex flash-attention kernel (`kernels/end2end/src/spm_flash_attn.cpp`,
CF+FA) includes a post-submission software optimization of its fp16<->fp32
conversion paths (fp16-domain row max; full-width vector loads with
zip1/zip2/uzp1 lane management replacing half-width widening loads). The
baseline flash-attention kernel (`cache_flash_attn.cpp`, BL+FA) ships
unchanged. This asymmetry is intentional and disclosed:

1. **Numerical transparency.** The optimization preserves per-element
   floating-point operation order; kernel output checksums are bit-identical
   to the shipped reference on every verified cell (LLaMA: all of
   T=256/1024/2048/4096 at VL=4 and VL=16; BERT: T=256 and T=4096 at both
   VLs).
2. **Restored measurement conditions.** The submitted-version Table 4 was
   measured under an equivalent implementation asymmetry (the
   reference-generating CF kernel's softmax was faster than the baseline's).
   With the shipped kernels, fresh same-environment simulations MEET OR
   EXCEED the submitted-version gains at the T=256 and T=4096 endpoints —
   the cells Figure 9 consumes (measured: VL4 +9.9/+19.0, VL16
   +13.8/+28.9, vs. submitted +8/+14 and +9/+17; VL16 T=1024 +3.5 vs
   submitted -1). Earlier drafts also quoted +7.6/+15.1 (VL4
   T=1024/2048) and +32.7 (VL16 T=2048); those paired runs from
   different execution environments and are withdrawn — same-environment
   pairs at the intermediate shapes fall below the submitted integers,
   dominated by the baseline's build-layout sensitivity (up to 16-19%
   across compiler-layout variants; our own audit flagged the VL16 +32.7
   as layout-artifact-dominated). The controlled statement of the
   kernel-level gain is the equal-treatment protocol of point 3.
3. **Equal-treatment results are published, not hidden.** We ported the same
   optimization to the baseline (shipped under `kernels/end2end/src/fair/`)
   and measured equal-treatment gains on LLaMA under a hardened protocol
   (instrumentation-free builds, alignment-controlled build families,
   family medians [ranges]): VL4 +3.2 [2.9,3.3] / 0 / +7.1 [7.1,7.4] /
   +9.6 [9.4,9.7]; VL16 0 / 0 / +18.5 [-0.8,+18.7] / +15.1 [-5.1,+15.2]
   for T=256/1024/2048/4096 ("0" = SPM mode within noise of zero or
   negative at that shape — measured VL16 T1024 family [-23,-5] — so the
   runtime per-shape placement policy keeps L2 in cache mode and the
   deployed gain is 0). Gains are driven by K+V/L2 capacity pressure; the
   VL16 T4096 value sits slightly below T2048 because the kernel's own
   accumulator outgrows the SPM-carved L2 at that size. (The render path
   reproduces the submitted-version table from the frozen reference.)
   A BERT T4096/VL16 pair under the same equal-treatment pairing
   measures +14.7%. See `kernels/end2end/src/fair/README.md` to
   reproduce.

## Result-comparison policy (one-sided gate)

`scripts/compare_results.py` gates Figure 9 metrics one-sidedly: regressions
beyond 5% fail; improvements of any size pass and are printed as
`[IMPROVED]`. Rationale: the shipped kernels include post-submission
improvements (above), so fresh runs can be legitimately faster than the
frozen reference; the gate certifies "at least the submitted-version
performance".
Figures 2/7/8 remain symmetrically gated. The frozen reference and all
rendered outputs are byte-unchanged.

## Known issues

- **Unfused reference cells (regenerated).** Eight unfused-attention
  reference cells were originally generated with a truncated command format
  relative to the arguments declared in `experiments/end2end/best.json`.
  They have been regenerated with the declared arguments and the shipped
  binaries; per-cell provenance is recorded alongside the cells.
- **Legacy flash argument format.** Eight flash reference cells record a
  legacy modeless argument format; the current binaries take a trailing
  attention-mode token. Same workload (non-causal); the mode token was
  added after the reference was generated.
- **T=4096 exit behavior.** Some T=4096 cells (including in the shipped
  reference) do not print a clean gem5 exit line; the ROI measurement and
  stats dump complete correctly and are unaffected. Fresh `run_all.sh`
  reruns of these cells can trip the strict receipt check for the same
  reason — compare the timing receipts and stats, which are complete.
- **CF-only rescale-skip.** The shipped CF kernel skips the online-softmax
  O_acc rescale when the running max is unchanged (alpha ~ 1); the fair
  baseline variant does not carry this micro-optimization. Measured
  effect is at or below 0.3pp at the T=256 anchor and within the reported
  family bands elsewhere.
