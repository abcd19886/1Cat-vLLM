// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <flashinfer/attention/sm70/volta_mma.cuh>

namespace fi = flashinfer::attention::sm70;
namespace {
constexpr int D = 128;
constexpr int BQ = 128;
constexpr int BK = 32;
constexpr int VLD = BK + 8;
constexpr int THREADS = (BQ / 16) * (BK / 16) * 32;
static_assert(THREADS * 8 == BK * D);
constexpr int shared_bytes() {
  constexpr int QLD = D + 8, PLD = BK + 8;
  return (BQ * QLD + BK * QLD + D * VLD + BQ * PLD) * 2 +
         (BQ * (BK / 16) * 2 + BQ * 2) * 4;
}

__global__ __launch_bounds__(THREADS,
                             1) void h3_noncausal(const half* q, const half* k,
                                                  const half* v, half* output,
                                                  int length, int heads,
                                                  float scale) {
  constexpr int QLD = D + 8, PLD = BK + 8;
  extern __shared__ __align__(32) unsigned char raw[];
  half* qs = reinterpret_cast<half*>(raw);
  half* ks = qs + BQ * QLD;
  half* vs = ks + BK * QLD;
  half* probabilities = vs + D * VLD;
  float* scores = reinterpret_cast<float*>(probabilities + BQ * PLD);
  float* maximum = scores + BQ * (BK / 16) * 2;
  float* denominator = maximum + BQ;
  const int tid = threadIdx.x, warp = tid / 32;
  const int warp_q = warp / (BK / 16), warp_k = warp % (BK / 16);
  const int lane = tid % 32;
  // Volta m16n16 accumulator element coordinates, matching the repository's
  // SM70 WMMA masking convention. SM70 is checked before dispatch.
  const int fragment_row =
      (lane & 1) + ((lane >> 2) & 1) * 8 + ((lane >> 4) & 1) * 4;
  const int fragment_col = ((lane >> 1) & 1) * 2 + ((lane >> 3) & 1) * 8;
  fi::AccumulatorFragment accumulators[D / BK];
  float register_alpha[2];
#pragma unroll
  for (int part = 0; part < D / BK; ++part)
    fi::init_accumulator_fragment(accumulators[part]);
  const int q_start = blockIdx.x * BQ;
  const int head = blockIdx.y % heads;
  const int batch = blockIdx.y / heads;
  const int64_t base = int64_t(batch) * length * heads * D + head * D;
  for (int i = tid; i < BQ * D; i += blockDim.x) {
    const int row = q_start + i / D;
    qs[(i / D) * QLD + i % D] = row < length
                                    ? q[base + int64_t(row) * heads * D + i % D]
                                    : __float2half(0.f);
  }
  if (tid < BQ) {
    maximum[tid] = -INFINITY;
    denominator[tid] = 0.f;
  }
  __syncthreads();
  for (int start = 0; start < length; start += BK) {
    if (start == 0) {
      for (int i = tid; i < BK * D; i += blockDim.x) {
        const int row = start + i / D;
        const int64_t position = base + int64_t(row) * heads * D + i % D;
        ks[(i / D) * QLD + i % D] =
            row < length ? k[position] : __float2half(0.f);
        vs[(i % D) * VLD + i / D] =
            row < length ? v[position] : __float2half(0.f);
      }
    }
    // Join both the initial loads and the previous iteration's prefetch.
    __syncthreads();
    fi::AccumulatorFragment qk;
    fi::init_accumulator_fragment(qk);
#pragma unroll
    for (int dim = 0; dim < D; dim += 16) {
      fi::AFragment qa;
      fi::QKBFragment kb;
      fi::load_a_fragment(qa, qs + warp_q * 16 * QLD + dim, QLD);
      fi::load_qk_b_fragment(kb, ks + warp_k * 16 * QLD + dim, QLD);
      fi::mma_sync_m16n16k16_row_col_f16f16f32(qk, qa, kb);
    }
    {
      // Volta distributes each accumulator row across lanes differing in
      // bits 1 and 3. Reduce its 16 columns in registers, then combine only
      // the BK/16 warp partials through shared memory.
      float row_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int i = 0; i < qk.num_elements; ++i) {
        const int col =
            warp_k * 16 + fragment_col + (i & 1) + ((i >> 2) & 1) * 4;
        qk.x[i] = start + col < length ? qk.x[i] * scale : -INFINITY;
        row_max[(i >> 1) & 1] = fmaxf(row_max[(i >> 1) & 1], qk.x[i]);
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        row_max[r] =
            fmaxf(row_max[r], __shfl_xor_sync(0xffffffff, row_max[r], 2));
        row_max[r] =
            fmaxf(row_max[r], __shfl_xor_sync(0xffffffff, row_max[r], 8));
        const int row = warp_q * 16 + fragment_row + r * 2;
        if ((lane & 10) == 0) scores[row * (BK / 16) + warp_k] = row_max[r];
      }
      __syncthreads();
      float new_max[2], row_sum[2] = {0.f, 0.f};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int row = warp_q * 16 + fragment_row + r * 2;
        new_max[r] = maximum[row];
#pragma unroll
        for (int w = 0; w < BK / 16; ++w)
          new_max[r] = fmaxf(new_max[r], scores[row * (BK / 16) + w]);
        register_alpha[r] = __expf(maximum[row] - new_max[r]);
      }
#pragma unroll
      for (int i = 0; i < qk.num_elements; ++i) {
        const int r = (i >> 1) & 1;
        const int row = warp_q * 16 + fragment_row + r * 2;
        const int col =
            warp_k * 16 + fragment_col + (i & 1) + ((i >> 2) & 1) * 4;
        const float p = __expf(qk.x[i] - new_max[r]);
        probabilities[row * PLD + col] = __float2half_rn(p);
        row_sum[r] += p;
      }
      float* partial_sums = scores + BQ * (BK / 16);
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 2);
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 8);
        const int row = warp_q * 16 + fragment_row + r * 2;
        if ((lane & 10) == 0)
          partial_sums[row * (BK / 16) + warp_k] = row_sum[r];
      }
      __syncthreads();
      if (warp_k == 0 && (lane & 10) == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int row = warp_q * 16 + fragment_row + r * 2;
          float sum = 0.f;
#pragma unroll
          for (int w = 0; w < BK / 16; ++w)
            sum += partial_sums[row * (BK / 16) + w];
          denominator[row] = denominator[row] * register_alpha[r] + sum;
          maximum[row] = new_max[r];
        }
      }
    }
    // Issue the next K/V global loads while the current V tile is consumed.
    union StagedVector {
      uint4 packed;
      half values[8];
    } next_k, next_v;
    const int next_row = start + BK + tid / (D / 8);
    const int next_col = (tid % (D / 8)) * 8;
    if (next_row < length) {
      const int64_t position = base + int64_t(next_row) * heads * D + next_col;
      if ((reinterpret_cast<uintptr_t>(k) % 16 == 0) &&
          (reinterpret_cast<uintptr_t>(v) % 16 == 0)) {
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(next_k.packed.x), "=r"(next_k.packed.y),
                       "=r"(next_k.packed.z), "=r"(next_k.packed.w)
                     : "l"(k + position));
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(next_v.packed.x), "=r"(next_v.packed.y),
                       "=r"(next_v.packed.z), "=r"(next_v.packed.w)
                     : "l"(v + position));
      } else {
        // Contiguous storage-offset views need scalar global loads.
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          next_k.values[j] = k[position + j];
          next_v.values[j] = v[position + j];
        }
      }
    } else {
      next_k.packed = make_uint4(0, 0, 0, 0);
      next_v.packed = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int part = 0; part < D / BK; ++part) {
      const int col = warp_k * (D * 16 / BK) + part * 16;
      const int row = warp_q * 16;
      auto& pv = accumulators[part];
#pragma unroll
      for (int i = 0; i < pv.num_elements; ++i)
        pv.x[i] *= register_alpha[(i >> 1) & 1];
#pragma unroll
      for (int kv = 0; kv < BK; kv += 16) {
        fi::AFragment pa;
        fi::QKBFragment vb;
        fi::load_a_fragment(pa, probabilities + row * PLD + kv, PLD);
        fi::load_qk_b_fragment(vb, vs + col * VLD + kv, VLD);
        fi::mma_sync_m16n16k16_row_col_f16f16f32(pv, pa, vb);
      }
    }
    __syncthreads();
    if (start + BK < length) {
      const int tile_row = tid / (D / 8);
      *reinterpret_cast<uint4*>(ks + tile_row * QLD + next_col) = next_k.packed;
      // Reuse dead P storage to transpose V cooperatively. Direct strided
      // stores from each thread's eight contiguous values collide heavily.
      static_assert(BQ * PLD >= BK * QLD);
      *reinterpret_cast<uint4*>(probabilities + tile_row * QLD + next_col) =
          next_v.packed;
      __syncthreads();
#pragma unroll
      for (int i = tid; i < D * BK; i += THREADS) {
        const int d = i / BK, kv = i % BK;
        vs[d * VLD + kv] = probabilities[kv * QLD + d];
      }
    }
  }
  {
#pragma unroll
    for (int part = 0; part < D / BK; ++part) {
      const auto& pv = accumulators[part];
#pragma unroll
      for (int i = 0; i < pv.num_elements; ++i) {
        const int row = warp_q * 16 + fragment_row + ((i >> 1) & 1) * 2;
        const int col = warp_k * (D * 16 / BK) + part * 16 + fragment_col +
                        (i & 1) + ((i >> 2) & 1) * 4;
        if (q_start + row < length)
          output[base + int64_t(q_start + row) * heads * D + col] =
              __float2half_rn(pv.x[i] / denominator[row]);
      }
    }
  }
}
}  // namespace

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                      double scale) {
  TORCH_CHECK(
      q.is_cuda() && k.device() == q.device() && v.device() == q.device(),
      "H3 attention requires same-device CUDA inputs");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 &&
                  k.scalar_type() == q.scalar_type() &&
                  v.scalar_type() == q.scalar_type(),
              "H3 FlashInfer-SM70 requires FP16");
  TORCH_CHECK(
      q.dim() == 4 && q.size(3) == D && q.size(1) > 0 &&
          q.sizes() == k.sizes() && q.sizes() == v.sizes(),
      "H3 FlashInfer-SM70 requires matching nonempty BSHD D128 MHA inputs");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "H3 FlashInfer-SM70 inputs must be contiguous");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto* properties = at::cuda::getDeviceProperties(q.get_device());
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "H3 FlashInfer kernel requires SM70");
  TORCH_CHECK(q.size(1) <= INT_MAX && q.size(0) * q.size(2) <= 65535,
              "H3 FlashInfer grid overflow");
  auto output = torch::empty_like(q);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      h3_noncausal, cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes()));
  dim3 grid((q.size(1) + BQ - 1) / BQ, q.size(0) * q.size(2));
  h3_noncausal<<<grid, (BQ / 16) * (BK / 16) * 32, shared_bytes(),
                 at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), q.size(1),
      q.size(2), float(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }
