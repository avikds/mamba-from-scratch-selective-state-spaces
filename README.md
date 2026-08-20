# Mamba from Scratch: Selective State Spaces

Implement the Mamba architecture from Gu and Dao 2023 as a tiny character-level language model in PyTorch. You will build RMSNorm, input-dependent Δ/B/C, log-parameterized A, paper-faithful zero-order-hold discretization, a sequential selective scan, the gated mixer, residual blocks, next-token training, and O(1) recurrent generation with a carried SSM state.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** rms_norm
- [x] **2.** silu
- [x] **3.** causal_depthwise_conv1d
- [x] **4.** in_proj_split
- [x] **5.** compute_delta
- [x] **6.** project_bc
- [x] **7.** make_diagonal_a
- [x] **8.** discretize_a_zoh
- [x] **9.** discretize_b_zoh
- [x] **10.** compare_euler_zoh_b
- [x] **11.** siso_state_update
- [x] **12.** scan_single_channel
- [x] **13.** selective_scan
- [x] **14.** compare_constant_vs_selective_delta
- [x] **15.** gate_scan_output
- [x] **16.** out_proj
- [x] **17.** mamba_mixer
- [x] **18.** mamba_block
- [x] **19.** run_mamba_lm_stack
- [x] **20.** mamba_lm_forward
- [x] **21.** next_token_cross_entropy
- [x] **22.** sgd_training_step
- [x] **23.** mamba_recurrent_step
- [x] **24.** greedy_generate
- [x] **25.** train_tiny_mamba_and_generate

---

Built on Deep-ML.
