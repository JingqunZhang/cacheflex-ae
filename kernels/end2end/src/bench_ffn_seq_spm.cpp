// This file is part of CacheFlex.  Its SVE FP16 GEMM microkernels are derived
// from the Arm Compute Library (https://github.com/ARM-software/ComputeLibrary),
// kernel sve_interleaved_fp16_mla_8x3VL.
// Upstream Copyright (c) Arm Limited.  SPDX-License-Identifier: MIT.
// Modifications Copyright (c) 2026 The CacheFlex Authors.
// See THIRD_PARTY_NOTICES for the full upstream copyright and permission notice.

/*
 * bench_ffn_seq_spm.cpp — Sequential FFN with SPM fused GEMM
 *
 * Same as bench_ffn_seq.cpp but GEMMs use spm_gemm_fused_8x3VL.
 * Activation (SwiGLU/GELU) uses inline SVE (not from kernels_sve.hpp).
 *
 * Usage: ./bench_ffn_seq_spm <model:0=llama,1=bert> <T> <MC> <KC> [n_iter]
 */

#include "kernels_spm.hpp"
#include "kernels_fused.hpp"
#include "layer_config.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Scalar prepack (can't include kernels_sve.hpp)
static size_t prepack_B_knm_size(size_t K, size_t N, size_t KC) {
    const size_t NT = 3 * svcnth();
    return ((K+KC-1)/KC) * ((N+NT-1)/NT) * KC * NT;
}
static void prepack_B_knm_spm(const __fp16* B, __fp16* Bp, size_t K, size_t N, size_t KC) {
    const size_t NT=3*svcnth(), Nt=(N+NT-1)/NT;
    for(size_t k0=0;k0<K;k0+=KC){size_t kc=std::min(KC,K-k0),kt=k0/KC;
        for(size_t n0=0;n0<N;n0+=NT){size_t nc=std::min(NT,N-n0),nt=n0/NT;
            __fp16*dst=Bp+(kt*Nt+nt)*KC*NT;
            for(size_t ki=0;ki<kc;++ki){
                for(size_t ni=0;ni<nc;++ni) dst[ki*NT+ni]=B[(k0+ki)*N+n0+ni];
                for(size_t ni=nc;ni<NT;++ni) dst[ki*NT+ni]=0;}
            for(size_t ki=kc;ki<KC;++ki)
                for(size_t ni=0;ni<NT;++ni) dst[ki*NT+ni]=0;}}
}

// Inline SVE activations (same as before)
static inline svfloat32_t sve_exp_f32_local(svbool_t pg, svfloat32_t x){
    x=svmax_n_f32_x(pg,x,-88.0f);svfloat32_t nf=svrintn_f32_x(pg,svmul_n_f32_x(pg,x,1.44269504088896f));
    svint32_t ni=svcvt_s32_f32_x(pg,nf);svfloat32_t r=svmls_n_f32_x(pg,x,nf,6.93147182e-1f);
    r=svmls_n_f32_x(pg,r,nf,-1.90465430e-9f);svfloat32_t p=svdup_n_f32(1.0f/120.0f);
    p=svmla_f32_x(pg,svdup_n_f32(1.0f/24.0f),r,p);p=svmla_f32_x(pg,svdup_n_f32(1.0f/6.0f),r,p);
    p=svmla_f32_x(pg,svdup_n_f32(0.5f),r,p);p=svmla_f32_x(pg,svdup_n_f32(1.0f),r,p);
    p=svmla_f32_x(pg,svdup_n_f32(1.0f),r,p);return svscale_f32_x(pg,p,ni);}

static void swiglu_fp16_sve(__fp16* gate, const __fp16* up, size_t n){
    const size_t V=svcntw();size_t i=0;
    for(;i+2*V<=n;i+=2*V){svbool_t pg=svptrue_b32();
        svbool_t a=svwhilelt_b16_u64(i,i+V),b=svwhilelt_b16_u64(i+V,i+2*V);
        svfloat32_t g0=svcvt_f32_f16_z(pg,svld1_f16(a,gate+i)),g1=svcvt_f32_f16_z(pg,svld1_f16(b,gate+i+V));
        svfloat32_t u0=svcvt_f32_f16_z(pg,svld1_f16(a,up+i)),u1=svcvt_f32_f16_z(pg,svld1_f16(b,up+i+V));
        svfloat32_t e0=sve_exp_f32_local(pg,svmin_n_f32_x(pg,svneg_f32_x(pg,g0),88.0f));
        svfloat32_t e1=sve_exp_f32_local(pg,svmin_n_f32_x(pg,svneg_f32_x(pg,g1),88.0f));
        svfloat32_t d0=svadd_n_f32_x(pg,e0,1.0f),d1=svadd_n_f32_x(pg,e1,1.0f);
        svfloat32_t r0=svrecpe_f32(d0),r1=svrecpe_f32(d1);
        r0=svmul_f32_x(pg,svrecps_f32(d0,r0),r0);r1=svmul_f32_x(pg,svrecps_f32(d1,r1),r1);
        svst1_f16(a,gate+i,svcvt_f16_f32_z(pg,svmul_f32_x(pg,svmul_f32_x(pg,g0,r0),u0)));
        svst1_f16(b,gate+i+V,svcvt_f16_f32_z(pg,svmul_f32_x(pg,svmul_f32_x(pg,g1,r1),u1)));}
    for(;i<n;i+=V){svbool_t pg=svwhilelt_b32_u64(i,n);svbool_t p16=svwhilelt_b16_u64(i,std::min(i+V,n));
        svfloat32_t g=svcvt_f32_f16_z(pg,svld1_f16(p16,gate+i)),u=svcvt_f32_f16_z(pg,svld1_f16(p16,up+i));
        svfloat32_t e=sve_exp_f32_local(pg,svmin_n_f32_x(pg,svneg_f32_x(pg,g),88.0f));
        svfloat32_t d=svadd_n_f32_x(pg,e,1.0f);svfloat32_t r=svrecpe_f32(d);r=svmul_f32_x(pg,svrecps_f32(d,r),r);
        svst1_f16(p16,gate+i,svcvt_f16_f32_z(pg,svmul_f32_x(pg,svmul_f32_x(pg,g,r),u)));}
}

static void gelu_fp16_inplace_sve(__fp16* buf, size_t n){
    const float S=1.702f;const size_t V=svcntw();size_t i=0;
    for(;i+2*V<=n;i+=2*V){svbool_t pg=svptrue_b32();
        svbool_t a=svwhilelt_b16_u64(i,i+V),b=svwhilelt_b16_u64(i+V,i+2*V);
        svfloat32_t x0=svcvt_f32_f16_z(pg,svld1_f16(a,buf+i)),x1=svcvt_f32_f16_z(pg,svld1_f16(b,buf+i+V));
        svfloat32_t e0=sve_exp_f32_local(pg,svmin_n_f32_x(pg,svneg_f32_x(pg,svmul_n_f32_x(pg,x0,S)),88.0f));
        svfloat32_t e1=sve_exp_f32_local(pg,svmin_n_f32_x(pg,svneg_f32_x(pg,svmul_n_f32_x(pg,x1,S)),88.0f));
        svfloat32_t d0=svadd_n_f32_x(pg,e0,1.0f),d1=svadd_n_f32_x(pg,e1,1.0f);
        svfloat32_t r0=svrecpe_f32(d0),r1=svrecpe_f32(d1);
        r0=svmul_f32_x(pg,svrecps_f32(d0,r0),r0);r1=svmul_f32_x(pg,svrecps_f32(d1,r1),r1);
        svst1_f16(a,buf+i,svcvt_f16_f32_z(pg,svmul_f32_x(pg,x0,r0)));
        svst1_f16(b,buf+i+V,svcvt_f16_f32_z(pg,svmul_f32_x(pg,x1,r1)));}
    for(;i<n;i+=V){svbool_t pg=svwhilelt_b32_u64(i,n);svbool_t p16=svwhilelt_b16_u64(i,std::min(i+V,n));
        svfloat32_t x=svcvt_f32_f16_z(pg,svld1_f16(p16,buf+i));
        svfloat32_t e=sve_exp_f32_local(pg,svmin_n_f32_x(pg,svneg_f32_x(pg,svmul_n_f32_x(pg,x,S)),88.0f));
        svfloat32_t d=svadd_n_f32_x(pg,e,1.0f);svfloat32_t r=svrecpe_f32(d);r=svmul_f32_x(pg,svrecps_f32(d,r),r);
        svst1_f16(p16,buf+i,svcvt_f16_f32_z(pg,svmul_f32_x(pg,x,r)));}
}

// V3 SPM fused GEMM. C rows use stride N_pad (multiple of the 3*VL tile
// width): the micro-kernel stores full tiles, so an unpadded stride would let
// tail tiles spill into the next row / past the allocation.
static void gemm_v3_spm_fused(
    const __fp16* A, const __fp16* B_pk, __fp16* C,
    size_t M, size_t K, size_t N, size_t MC, size_t KC, size_t N_pad)
{
    const size_t MT=8, NT=3*svcnth();
    const size_t N_tiles=(N+NT-1)/NT, max_ab=(MC+MT-1)/MT;
    AlignedBuffer<__fp16> A_mc(max_ab*MT*KC);

    for(size_t k0=0;k0<K;k0+=KC){
        size_t kc=std::min(KC,K-k0), kt=k0/KC;
        for(size_t m0=0;m0<M;m0+=MC){
            size_t mc=std::min(MC,M-m0), ablocks=(mc+MT-1)/MT;
            pack_A_fp16_8row(A,K,A_mc.data(),M,m0,K,k0,mc,kc);
            for(size_t n0=0;n0<N;n0+=NT){
                size_t nt=n0/NT;
                pack_B_tile_to_spm(B_pk+(kt*N_tiles+nt)*KC*NT, NT, kc, KC, 0);
                spm_gemm_fused_8x3VL(A_mc.data(), C+m0*N_pad+n0,
                    (int)ablocks,(int)kc,(int)N_pad,(k0==0)?1:0,
#if defined(VL_8) || defined(VL_16)
                    KC,   // KC-offset addressing (same pattern as bench_kernel_spm.cpp)
#endif
                    0);
            }
        }
    }
}

int main(int argc, char** argv)
{
    if(argc<5){fprintf(stderr,"Usage: %s <model:0/1> <T> <MC> <KC> [n_iter]\n",argv[0]);return 1;}
    const bool is_llama=(atoi(argv[1])==0);
    const size_t M=(size_t)atoi(argv[2]);
    const size_t K=is_llama?LlamaConfig::D:BertConfig::D;
    const size_t N=is_llama?LlamaConfig::FFN_dim:BertConfig::FFN_dim;
    const size_t MC=(size_t)atoi(argv[3]),KC=(size_t)atoi(argv[4]);
    const int n_iter=(argc>5)?atoi(argv[5]):1;
    const char*mdl=is_llama?"llama":"bert";

    AlignedBuffer<__fp16> X(M*K),output(M*N);
    fill_random(X.data(),M*K,-0.1f,0.1f,1);
    AlignedBuffer<__fp16> Wg_raw(K*N),Wg_pk(prepack_B_knm_size(K,N,KC));
    fill_random(Wg_raw.data(),K*N,-0.02f,0.02f,2);
    prepack_B_knm_spm(Wg_raw.data(),Wg_pk.data(),K,N,KC);
    AlignedBuffer<__fp16> Wu_raw(is_llama?K*N:1),Wu_pk(is_llama?prepack_B_knm_size(K,N,KC):1);
    if(is_llama){fill_random(Wu_raw.data(),K*N,-0.02f,0.02f,3);prepack_B_knm_spm(Wu_raw.data(),Wu_pk.data(),K,N,KC);}

    const size_t NT=3*svcnth();
    const size_t N_pad=((N+NT-1)/NT)*NT;  // full-tile row stride
    AlignedBuffer<__fp16> gate_buf(M*N_pad),up_buf(is_llama?M*N_pad:1);

    auto run=[&]()->double{
        auto t0=Clock::now();
        if(is_llama){
            gemm_v3_spm_fused(X.data(),Wg_pk.data(),gate_buf.data(),M,K,N,MC,KC,N_pad);
            gemm_v3_spm_fused(X.data(),Wu_pk.data(),up_buf.data(),M,K,N,MC,KC,N_pad);
            for(size_t r=0;r<M;++r)
                swiglu_fp16_sve(gate_buf.data()+r*N_pad,up_buf.data()+r*N_pad,N);
        } else {
            gemm_v3_spm_fused(X.data(),Wg_pk.data(),gate_buf.data(),M,K,N,MC,KC,N_pad);
            for(size_t r=0;r<M;++r)
                gelu_fp16_inplace_sve(gate_buf.data()+r*N_pad,N);
        }
        return us_since(t0);};

    run();
    double total_us=0;
#ifdef GEM5
    m5_reset_stats(0, 0);
#endif
    ROI_BEGIN();
    for(int i=0;i<n_iter;++i) total_us+=run();
    ROI_END();
#ifdef GEM5
    m5_dump_stats(0, 0);
#endif
    double avg=total_us/n_iter;int ng=is_llama?2:1;
    printf("RESULT kernel=ffn_seq_spm model=%s T=%zu M=%zu K=%zu N=%zu MC=%zu KC=%zu time_us=%.2f gflops=%.2f\n",
           mdl,M,M,K,N,MC,KC,avg,2.0*ng*M*(double)K*N*1e-9/(avg*1e-6));
    { double _ck=0; const __fp16* _p=gate_buf.data();
      for (size_t _r=0;_r<M;++_r)
        for (size_t _c=0;_c<N;++_c) _ck += (double)_p[_r*N_pad+_c];
      printf("CHECKSUM: %.9f\n", _ck); }
    return 0;
}
