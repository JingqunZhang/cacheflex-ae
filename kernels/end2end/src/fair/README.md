# Equal-treatment ("fair") baseline variant

`cache_flash_attn_fair.cpp` is the shipped baseline flash-attention kernel
with the SAME fp16<->fp32 conversion-path optimization that the shipped
CacheFlex kernel (`../spm_flash_attn.cpp`) carries (fp16-domain row max;
full-width loads with zip1/zip2/uzp1 lane management). Its outputs are
bit-identical to the shipped baseline (checksum-verified on every cell).
One CF-only micro-optimization is intentionally NOT ported: the CF kernel
skips the online-softmax O_acc rescale when the running max is unchanged
(<= 0.3pp at the T=256 anchor; disclosed in docs/CLAIMS.md Known issues).

It is provided so the equal-treatment comparison disclosed in docs/CLAIMS.md
can be reproduced independently. First `source setup_env.sh` (exports
CROSS_CXX, CACHEFLEX_ROOT, M5OP_OBJ; m5op.o is built in SETUP.md step 6),
then build exactly like the shipped baseline:

    $CROSS_CXX -O3 -std=c++17 -march=armv8.2-a+sve+fp16 -static -Isrc \
        -DGEM5 -DGEM5_SE -I$CACHEFLEX_ROOT/gem5/include \
        src/fair/cache_flash_attn_fair.cpp $M5OP_OBJ -lm \
        -o bin/cache_flash_attn_fair

Equal-treatment gains measured with this variant on LLaMA (both kernels
optimized, instrumentation-free builds, alignment-controlled build
families; family medians [ranges]): VL4 +3.2 [2.9,3.3] / 0 /
+7.1 [7.1,7.4] / +9.6 [9.4,9.7] and VL16 0 / 0 / +18.5 [-0.8,+18.7] /
+15.1 [-5.1,+15.2] for T=256/1024/2048/4096 ("0" = SPM mode within noise
of zero or negative at that shape — measured VL16 T1024 family [-23,-5] —
so the deployed per-shape placement policy keeps L2 in cache mode and the
deployed gain is 0).
