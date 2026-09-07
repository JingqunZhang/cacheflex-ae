/*
 * 08_ffn_swiglu.cpp — LLaMA SwiGLU Feed-Forward Network
 *
 * SwiGLU FFN:
 *   gate[T, FFN_dim] = x @ W_gate   (T,D) × (D,FFN_dim)
 *   up  [T, FFN_dim] = x @ W_up     (T,D) × (D,FFN_dim)
 *   hidden = SiLU(gate) * up         element-wise: SiLU(g)·u
 *   out [T, D]       = hidden @ W_down  (T,FFN_dim) × (FFN_dim,D)
 *
 * vs BERT GELU-FFN: 3 GEMMs instead of 2 (gate + up + down)
 *
 * LLaMA-3.2-1B: T=512, D=2048, FFN_dim=8192
 * FLOPs: 3 × 2×T×D×FFN_dim = 3 × 2×512×2048×8192 ≈ 51.5 GFLOPs
 *
 * Usage: ./08_ffn_swiglu [T] [D] [FFN_dim] [n_iter]
 *        default: 512 2048 8192 5
 */
#include "common.hpp"
#include "kernels_sve.hpp"

int main(int argc, char** argv)
{
    const int    T       = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int    D       = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int    FFN_dim = (argc > 3) ? std::atoi(argv[3]) : 8192;
    const int    N_ITER  = (argc > 4) ? std::atoi(argv[4]) : 5;
    const size_t NT_hw  = 3 * svcnth();
    const size_t KC_arg = (argc > 5) ? (size_t)std::atoi(argv[5]) : 0;
    const size_t KC     = KC_arg ? KC_arg : 24576 / NT_hw;  // default: B-panel ≈ 48KB (~¾ L1)
    const size_t MC_arg = (argc > 6) ? (size_t)std::atoi(argv[6]) : 0;
    const size_t MC     = MC_arg ? MC_arg : 16 * svcnth();

    const double W_gate_MB = (double)D * FFN_dim * 2.0 / 1048576;
    const double W_up_MB   = (double)D * FFN_dim * 2.0 / 1048576;
    const double W_dn_MB   = (double)FFN_dim * D * 2.0 / 1048576;
    const double in_MB     = (double)T * D       * 2.0 / 1048576;
    const double hid_MB    = (double)T * FFN_dim * 2.0 / 1048576;

    const size_t pk_gate_sz = prepack_B_knm_size(D,       FFN_dim, KC);
    const size_t pk_up_sz   = prepack_B_knm_size(D,       FFN_dim, KC);
    const size_t pk_dn_sz   = prepack_B_knm_size(FFN_dim, D,       KC);

    printf("=== 08 FFN SwiGLU ===  T=%d  D=%d  FFN_dim=%d  n_iter=%d\n",
           T, D, FFN_dim, N_ITER);
    printf("  input   [%d, %d]        %.2f MB\n", T, D, in_MB);
    printf("  W_gate  [%d, %d]   %.2f MB raw\n", D, FFN_dim, W_gate_MB);
    printf("  W_up    [%d, %d]   %.2f MB raw\n", D, FFN_dim, W_up_MB);
    printf("  hidden  [%d, %d]  %.2f MB  (gate+up merged, after SwiGLU)\n", T, FFN_dim, hid_MB);
    printf("  W_down  [%d, %d]   %.2f MB raw\n", FFN_dim, D, W_dn_MB);
    printf("  output  [%d, %d]        %.2f MB\n\n", T, D, in_MB);

    AlignedBuffer<__fp16> x_in((size_t)T * D);
    AlignedBuffer<__fp16> W_gate((size_t)D * FFN_dim),   W_gate_pk(pk_gate_sz);
    AlignedBuffer<__fp16> W_up((size_t)D * FFN_dim),     W_up_pk(pk_up_sz);
    AlignedBuffer<__fp16> W_down((size_t)FFN_dim * D),   W_down_pk(pk_dn_sz);
    AlignedBuffer<__fp16> gate((size_t)T * FFN_dim);   // gate GEMM output + in-place SwiGLU
    AlignedBuffer<__fp16> up((size_t)T * FFN_dim);     // up GEMM output
    AlignedBuffer<__fp16> out((size_t)T * D);

    fill_random(x_in.data(),   (size_t)T*D,        -0.1f,  0.1f,  1);
    fill_random(W_gate.data(), (size_t)D*FFN_dim,  -0.02f, 0.02f, 2);
    fill_random(W_up.data(),   (size_t)D*FFN_dim,  -0.02f, 0.02f, 3);
    fill_random(W_down.data(), (size_t)FFN_dim*D,  -0.02f, 0.02f, 4);

    prepack_B_knm(W_gate.data(), W_gate_pk.data(), D,       FFN_dim, KC);
    prepack_B_knm(W_up.data(),   W_up_pk.data(),   D,       FFN_dim, KC);
    prepack_B_knm(W_down.data(), W_down_pk.data(), FFN_dim, D,       KC);

    // --- COLD: flush all weight packs to DRAM ---
    printf("--- COLD (W_gate_pk + W_up_pk + W_down_pk flushed to DRAM, gem5 ROI) ---\n");
    flush_dcache_range(W_gate_pk.data(), pk_gate_sz * sizeof(__fp16));
    flush_dcache_range(W_up_pk.data(),   pk_up_sz   * sizeof(__fp16));
    flush_dcache_range(W_down_pk.data(), pk_dn_sz   * sizeof(__fp16));

    ROI_BEGIN();
    {
        double fg = 2.0*T*D*FFN_dim, fd = 2.0*T*FFN_dim*D;
        auto t0 = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_gate_pk.data(), gate.data(), T, FFN_dim, D, MC, KC);
        double ug = us_since(t0);
        printf("  W_gate GEMM: %7.1f us   %.2f GFLOPS\n", ug, fg/(ug*1e3));
        t0 = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_up_pk.data(), up.data(), T, FFN_dim, D, MC, KC);
        double uu = us_since(t0);
        printf("  W_up   GEMM: %7.1f us   %.2f GFLOPS\n", uu, fg/(uu*1e3));
        t0 = Clock::now();
        swiglu_fp16(gate.data(), up.data(), (size_t)T * FFN_dim);
        printf("  SwiGLU     : %7.1f us\n", us_since(t0));
        t0 = Clock::now();
        gemm_knm_prepacked(gate.data(), W_down_pk.data(), out.data(), T, D, FFN_dim, MC, KC);
        double ud = us_since(t0);
        printf("  W_down GEMM: %7.1f us   %.2f GFLOPS\n", ud, fd/(ud*1e3));
    }
    ROI_END();
    printf("\n");

    // --- WARM ---
    printf("--- WARM (weights in cache) ---\n");
    for (int i = 0; i < (N_ITER > 1 ? 1 : 0); ++i) {
        gemm_knm_prepacked(x_in.data(), W_gate_pk.data(), gate.data(), T, FFN_dim, D, MC, KC);
        gemm_knm_prepacked(x_in.data(), W_up_pk.data(),   up.data(),   T, FFN_dim, D, MC, KC);
        swiglu_fp16(gate.data(), up.data(), (size_t)T * FFN_dim);
        gemm_knm_prepacked(gate.data(), W_down_pk.data(), out.data(), T, D, FFN_dim, MC, KC);
    }

    double tg=0, tu=0, ts=0, td=0;
    GemmTiming timG, timU, timD;
    for (int i = 0; i < N_ITER; ++i) {
        auto tp = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_gate_pk.data(), gate.data(), T, FFN_dim, D, MC, KC, &timG);
        tg += us_since(tp);
        tp = Clock::now();
        gemm_knm_prepacked(x_in.data(), W_up_pk.data(), up.data(), T, FFN_dim, D, MC, KC, &timU);
        tu += us_since(tp);
        tp = Clock::now();
        swiglu_fp16(gate.data(), up.data(), (size_t)T * FFN_dim);
        ts += us_since(tp);
        tp = Clock::now();
        gemm_knm_prepacked(gate.data(), W_down_pk.data(), out.data(), T, D, FFN_dim, MC, KC, &timD);
        td += us_since(tp);
    }
    tg/=N_ITER; tu/=N_ITER; ts/=N_ITER; td/=N_ITER;
    timG.t_packA/=N_ITER; timG.t_compute/=N_ITER; timG.t_writeback/=N_ITER;
    timU.t_packA/=N_ITER; timU.t_compute/=N_ITER; timU.t_writeback/=N_ITER;
    timD.t_packA/=N_ITER; timD.t_compute/=N_ITER; timD.t_writeback/=N_ITER;
    double fg = 2.0*T*D*FFN_dim, fd = 2.0*T*FFN_dim*D;
    printf("  W_gate GEMM: %7.1f us  [packA %5.1f  compute %6.1f  wb %5.1f]  %.2f GFLOPS\n",
           tg, timG.t_packA, timG.t_compute, timG.t_writeback, fg/(tg*1e3));
    printf("  W_up   GEMM: %7.1f us  [packA %5.1f  compute %6.1f  wb %5.1f]  %.2f GFLOPS\n",
           tu, timU.t_packA, timU.t_compute, timU.t_writeback, fg/(tu*1e3));
    printf("  SwiGLU     : %7.1f us\n", ts);
    printf("  W_down GEMM: %7.1f us  [packA %5.1f  compute %6.1f  wb %5.1f]  %.2f GFLOPS\n",
           td, timD.t_packA, timD.t_compute, timD.t_writeback, fd/(td*1e3));
    printf("  ──────────────────────────────────────────\n");
    double tot = tg+tu+ts+td;
    printf("  total       : %7.1f us   %.2f GFLOPS combined\n\n", tot, (2*fg+fd)/(tot*1e3));

    printf("  W_gate+W_up packed: %.2f+%.2f MB; W_down packed: %.2f MB\n",
           pk_gate_sz*2.0/1048576, pk_up_sz*2.0/1048576, pk_dn_sz*2.0/1048576);
    printf("  Total packed weights: %.2f MB — dominates L2/L3 pressure\n",
           (pk_gate_sz+pk_up_sz+pk_dn_sz)*2.0/1048576);
    printf("  hidden[T,FFN_dim]=%.2f MB passes through between W_up and W_down\n", hid_MB);
    return 0;
}
