#include "silu.h"
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FFN_HAVE_NEON 1
#else
#define FFN_HAVE_NEON 0
#endif

namespace ffn {

float silu_scalar_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

#if FFN_HAVE_NEON

// Cephes-derived range-reduction + degree-5 polynomial exp(x), vectorized.
// exp(x) = 2^n * exp(r), x = n*ln2 + r, |r| <= ln2/2. n is reconstructed by
// direct IEEE-754 exponent-bit manipulation (no branches, no libm calls).
// Accuracy: ~1e-6 relative error over the reduced range — this is the
// "<0.5% numerical error" budget almost entirely consumed by quantization
// downstream, not by this approximation.
static inline float32x4_t exp_neon_ps(float32x4_t x) {
    const float32x4_t exp_hi = vdupq_n_f32(88.3762626647950f);
    const float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    x = vminq_f32(x, exp_hi);
    x = vmaxq_f32(x, exp_lo);

    const float32x4_t log2ef = vdupq_n_f32(1.44269504088896341f);
    float32x4_t fx = vmlaq_f32(vdupq_n_f32(0.5f), x, log2ef);

    // floor(fx) via truncate-then-correct (no vrndmq_f32 dependency).
    int32x4_t emm0 = vcvtq_s32_f32(fx);
    float32x4_t tmp = vcvtq_f32_s32(emm0);
    uint32x4_t gt_mask = vcgtq_f32(tmp, fx);
    float32x4_t correction = vreinterpretq_f32_u32(
        vandq_u32(gt_mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f))));
    fx = vsubq_f32(tmp, correction);

    // Two-part ln(2) subtraction for precision (single-precision ln2 alone
    // loses too many bits of r near large x).
    x = vsubq_f32(x, vmulq_n_f32(fx, 0.693359375f));
    x = vsubq_f32(x, vmulq_n_f32(fx, -2.12194440e-4f));

    const float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vmlaq_f32(x, y, z);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));

    // Reconstruct 2^n by writing n directly into the float's exponent field.
    emm0 = vcvtq_s32_f32(fx);
    emm0 = vaddq_s32(emm0, vdupq_n_s32(127));
    emm0 = vshlq_n_s32(emm0, 23);
    const float32x4_t pow2n = vreinterpretq_f32_s32(emm0);

    return vmulq_f32(y, pow2n);
}

float32x4_t silu_neon4(float32x4_t x) {
    float32x4_t neg_x = vnegq_f32(x);
    float32x4_t e = exp_neon_ps(neg_x);
    float32x4_t denom = vaddq_f32(e, vdupq_n_f32(1.0f));
    float32x4_t sig = vdivq_f32(vdupq_n_f32(1.0f), denom); // aarch64 has hw fdiv
    return vmulq_f32(x, sig);
}

void silu_inplace(const float* in, float* out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, silu_neon4(vld1q_f32(in + i)));
    }
    for (; i < n; ++i) out[i] = silu_scalar_ref(in[i]);
}

#else // portable fallback

void silu_inplace(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = silu_scalar_ref(in[i]);
}

#endif

} // namespace ffn
