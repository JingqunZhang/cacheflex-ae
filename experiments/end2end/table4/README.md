# Table 4 (tab:flash_scaling) reference data

Table 4 reports the measured CacheFlex SPM performance change of the
FlashAttention kernel — **CF+FA vs. BL+FA on LLaMA** — across per-head
`K+V` working-set sizes, at VL=4 and VL=16. Each entry is

```
gain = (t_BL+FA_cold - t_CF+FA_cold) / t_CF+FA_cold
```

where `t_BL+FA` is the cold-cache single-iteration time of the
cache-resident FlashAttention baseline (`cache_flash_attn`) and
`t_CF+FA` is the SPM CacheFlex variant (`spm_flash_attn`).

## Where the eight points come from

| T | K+V | VL=4 (cflash / sflash) | VL=16 (cflash / sflash) | source |
|---:|---:|---|---|---|
| 256  | 64 KB   | `br64_bc576` / `br64_bc384` | `br64_bc256` / `br32_bc256` | Figure 9 reference |
| 1024 | 256 KB  | `br64_bc576` / `br32_bc576` | `br64_bc384` / `br64_bc384` | this pool |
| 2048 | 512 KB  | `br64_bc576` / `br64_bc576` | `br64_bc384` / `br64_bc384` | this pool |
| 4096 | 1024 KB | `br64_bc576` / `br64_bc576` | `br64_bc384` / `br32_bc384` | Figure 9 reference |

The `T=256` and `T=4096` endpoints are the **exact same flash cells that
Figure 9 consumes**; the renderer reads them straight from the frozen
end-to-end reference (`../results/reference/simulation_results/{vl4,vl16}`),
so the Table 4 endpoints and the Figure 9 flash bars can never disagree.

The intermediate `T=1024` and `T=2048` points are shipped here, in a
self-contained companion pool, so Table 4 can be regenerated with no gem5
simulation while leaving the frozen Figure 9 108-cell contract untouched.
Every cell was produced under the same paper CPU configuration as the
frozen reference (identical L2/L3 geometry, cache/SPM ports, ROB/LQ/SQ,
prefetchers, and DRAM timing; only `sve_vl_se` differs 4↔16 by block) —
`plot_table4_flash.py --check` re-verifies this on every run.

## Regenerate / verify (no simulation)

```
python3 ../scripts/plot_table4_flash.py --check
```

This parses the cold-run timing receipts, asserts each cell recorded the
paper configuration and exited cleanly, writes `table4_data.json`,
`table4_flash.tex`, and `table4_flash.md`, and asserts the rendered gains
reproduce the submitted-version Table 4 (+8/+3/+10/+14 at VL=4,
+9/-1/+22/+17 at VL=16) within 1.5 percentage points. (Equal-treatment
results under a hardened protocol are documented in `docs/CLAIMS.md`
point 3.) It is wired into `scripts/render_reference.sh` as step [5/5].
