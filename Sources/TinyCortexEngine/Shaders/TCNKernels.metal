#include <metal_stdlib>
using namespace metal;

// Data layout: [T, C] (sample-major) for coalesced access.
// Each thread handles one sample (or one element for PReLU).

/// 1×1 convolution: output[t][o] = bias[o] + sum_i weight[o * in_ch + i] * input[t * in_ch + i]
/// Used for input conv (1→C) and output conv (C→1).
kernel void conv1x1(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    device const float* weight  [[buffer(2)]],  // [out_ch, in_ch]
    device const float* bias    [[buffer(3)]],  // [out_ch]
    constant     uint&  T       [[buffer(4)]],
    constant     uint&  in_ch   [[buffer(5)]],
    constant     uint&  out_ch  [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= T) return;

    for (uint o = 0; o < out_ch; o++) {
        float sum = bias[o];
        for (uint i = 0; i < in_ch; i++) {
            sum += weight[o * in_ch + i] * input[tid * in_ch + i];
        }
        output[tid * out_ch + o] = sum;
    }
}

/// Causal dilated convolution with kernel_size=3.
/// State is a ring buffer indexed by `head`. When t - lookback < 0, reads from state.
/// PyTorch Conv1d cross-correlation with causal padding:
///   output[o][t] = bias[o] + sum_{i,k} weight[o][i][k] * input[i][t - (2-k)*dilation]
/// Weight layout: [out_ch, in_ch, 3] row-major.
kernel void causal_dilated_conv3(
    device const float* input   [[buffer(0)]],  // [T, C] current activation
    device       float* output  [[buffer(1)]],  // [T, C]
    device const float* state   [[buffer(2)]],  // [state_len, C] ring buffer
    device const float* weight  [[buffer(3)]],  // [C, C, 3]
    device const float* bias    [[buffer(4)]],  // [C]
    constant     uint&  T       [[buffer(5)]],
    constant     uint&  C       [[buffer(6)]],
    constant     uint&  dilation [[buffer(7)]],
    constant     uint&  state_len [[buffer(8)]],
    constant     uint&  head     [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= T) return;

    int t = (int)tid;
    int sl = (int)state_len;

    for (uint o = 0; o < C; o++) {
        float sum = bias[o];
        for (uint i = 0; i < C; i++) {
            // k=0: lookback = 2*dilation, k=1: dilation, k=2: 0
            for (uint k = 0; k < 3; k++) {
                int lookback = (int)(2 - k) * (int)dilation;
                int src_t = t - lookback;
                float val;
                if (src_t < 0) {
                    // Ring buffer: logical index = state_len + src_t
                    uint logical = (uint)(sl + src_t);
                    uint phys = (head + logical) % state_len;
                    val = state[phys * C + i];
                } else {
                    val = input[src_t * C + i];
                }
                sum += weight[o * C * 3 + i * 3 + k] * val;
            }
        }
        output[t * C + o] = sum;
    }
}

/// PReLU: y = x >= 0 ? x : alpha * x. In-place. One thread per element.
kernel void prelu(
    device       float* data    [[buffer(0)]],  // [T, C] in-place
    device const float* alpha   [[buffer(1)]],  // [C]
    constant     uint&  count   [[buffer(2)]],  // T * C
    constant     uint&  C       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    uint c = tid % C;
    float x = data[tid];
    data[tid] = x >= 0.0f ? x : alpha[c] * x;
}

/// 1×1 conv + residual add (fused).
/// output[t][o] = residual[t][o] + bias[o] + sum_i weight[o * C + i] * input[t * C + i]
/// output and residual may alias (same buffer). Safe because each thread handles one sample.
kernel void conv1x1_residual(
    device const float* input    [[buffer(0)]],  // [T, C] conv input (activationB)
    device const float* residual [[buffer(1)]],  // [T, C] block input (activationA)
    device       float* output   [[buffer(2)]],  // [T, C] result (same buffer as residual)
    device const float* weight   [[buffer(3)]],  // [C, C]
    device const float* bias     [[buffer(4)]],  // [C]
    constant     uint&  T        [[buffer(5)]],
    constant     uint&  C        [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= T) return;

    // Read all residual values first since output may alias residual
    float res[64];
    for (uint o = 0; o < C && o < 64; o++) {
        res[o] = residual[tid * C + o];
    }

    for (uint o = 0; o < C; o++) {
        float sum = bias[o];
        for (uint i = 0; i < C; i++) {
            sum += weight[o * C + i] * input[tid * C + i];
        }
        output[tid * C + o] = res[o] + sum;
    }
}

/// Write activation samples into state ring buffer, overwriting oldest entries.
/// When state_len <= T: writes last state_len samples of activation.
/// When state_len > T: writes all T samples at head position.
kernel void update_state(
    device const float* activation [[buffer(0)]],  // [T, C]
    device       float* state      [[buffer(1)]],  // [state_len, C] ring buffer
    constant     uint&  T          [[buffer(2)]],
    constant     uint&  C          [[buffer(3)]],
    constant     uint&  state_len  [[buffer(4)]],
    constant     uint&  head       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint write_count = min(T, state_len);
    if (tid >= write_count * C) return;
    uint i = tid / C;
    uint c = tid % C;
    // Source: last write_count samples of activation (or all T if T < state_len)
    uint src_t = T - write_count + i;
    // Destination: ring buffer starting at head
    uint phys = (head + i) % state_len;
    state[phys * C + c] = activation[src_t * C + c];
}
