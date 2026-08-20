"""
Mamba from Scratch: Selective State Spaces

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - rms_norm
import torch

def rms_norm(x, weight, eps=1e-5):
    """Normalize a hidden sequence with RMSNorm using a learned per-channel scale."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# Step 2 - silu
def silu(x):
    """Apply the SiLU activation elementwise."""
    return x * torch.sigmoid(x)

# Step 3 - causal_depthwise_conv1d
import torch.nn.functional as F

def causal_depthwise_conv1d(x, weight, bias=None):
    """Run a causal depthwise 1-D convolution over a (B, L, E) sequence.

    Args:
        x: (B, L, E) input sequence.
        weight: (E, K) per-channel kernel.
        bias: optional (E,) added after the convolution.

    Returns:
        (B, L, E) output sequence.
    """
    B, L, E = x.shape
    K = weight.shape[1]

    # Convert (B, L, E) -> (B, E, L)
    x = x.transpose(1, 2)

    # Left-pad by exactly K - 1 positions for causal convolution.
    x = F.pad(x, (K - 1, 0))

    # Conv1d expects weights of shape (out_channels, in_channels/groups, K).
    # With groups=E, each channel is convolved independently.
    weight = weight.unsqueeze(1)

    y = F.conv1d(
        x,
        weight,
        bias=bias,
        stride=1,
        padding=0,
        groups=E,
    )

    # Convert (B, E, L) -> (B, L, E)
    return y.transpose(1, 2)

# Step 4 - in_proj_split
def in_proj_split(u, weight, bias=None):
    """Project tokens to expanded inner width and split into SSM input x and gate z."""
    projected = u @ weight.transpose(-1, -2)

    if bias is not None:
        projected = projected + bias

    x, z = torch.chunk(projected, 2, dim=-1)

    return x, z

# Step 5 - compute_delta
def compute_delta(x, weight, bias=None):
    """Compute a strictly positive per-token timestep Delta.

    x: (B, L, E), weight: (E, E) nn.Linear layout, bias: optional (E,).
    Returns delta of shape (B, L, E).
    """
    projected = x @ weight.transpose(-1, -2)

    if bias is not None:
        projected = projected + bias

    return torch.nn.functional.softplus(projected)

# Step 6 - project_bc
def project_bc(x, weight_b, weight_c):
    """Project the SSM input to input-dependent B and C state vectors of size N."""
    B_ssm = x @ weight_b.transpose(-1, -2)
    C_ssm = x @ weight_c.transpose(-1, -2)

    return B_ssm, C_ssm

# Step 7 - make_diagonal_a
def make_diagonal_a(log_a):
    """Map unconstrained log-A of shape (E, N) to a strictly negative diagonal A."""
    return -torch.exp(log_a)

# Step 8 - discretize_a_zoh
def discretize_a_zoh(delta, a):
    """Discretize a diagonal continuous state matrix with zero-order hold.

    delta: torch tensor of shape (..., d)
    a: torch tensor of shape (d, n)
    Returns a_bar of shape (..., d, n).
    """
    return torch.exp(delta.unsqueeze(-1) * a)

# Step 9 - discretize_b_zoh
def discretize_b_zoh(delta, a, b):
    """Discretize B with the exact diagonal zero-order-hold formula.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        b_bar: (batch, seq_len, d_inner, d_state) discrete B.
    """
    delta_a = delta.unsqueeze(-1) * a
    b_factor = (torch.exp(delta_a) - 1.0) / a

    return b_factor * b.unsqueeze(-2)

# Step 10 - compare_euler_zoh_b
def compare_euler_zoh_b(delta, a, b):
    """Compare exact ZOH discrete B to the Euler shortcut.

    Args:
        delta: (batch, seq_len, d_inner) timesteps.
        a: (d_inner, d_state) continuous diagonal A (strictly negative).
        b: (batch, seq_len, d_state) continuous input-dependent B.

    Returns:
        dict with keys 'b_bar_zoh', 'b_bar_euler', and 'abs_diff', each
        of shape (batch, seq_len, d_inner, d_state).
    """
    b_bar_zoh = discretize_b_zoh(delta, a, b)

    b_bar_euler = delta.unsqueeze(-1) * b.unsqueeze(2)

    abs_diff = torch.abs(b_bar_zoh - b_bar_euler)

    return {
        "b_bar_zoh": b_bar_zoh,
        "b_bar_euler": b_bar_euler,
        "abs_diff": abs_diff,
    }

# Step 11 - siso_state_update
def siso_state_update(h_prev, a_bar, b_bar, c, x_t):
    """Apply one SISO state update and return the scalar readout."""
    h_t = a_bar * h_prev + b_bar * x_t
    y_t = (c * h_t).sum()

    return y_t, h_t

# Step 12 - scan_single_channel
def scan_single_channel(x, a_bar, b_bar, c, h0=None):
    """Scan a single channel sequentially over time and return both the outputs and the final hidden state."""
    L = x.shape[0]
    N = a_bar.shape[-1]

    if h0 is None:
        h = torch.zeros(N, dtype=x.dtype, device=x.device)
    else:
        h = h0

    y = torch.empty(L, dtype=x.dtype, device=x.device)

    for t in range(L):
        h = a_bar[t] * h + b_bar[t] * x[t]
        y[t] = (c[t] * h).sum()

    return y, h

# Step 13 - selective_scan
def selective_scan(x, a_bar, b_bar, c, h0=None):
    """Run a selective scan over a batched multi-channel sequence."""
    B, L, E = x.shape
    N = a_bar.shape[-1]

    if h0 is None:
        h = torch.zeros(B, E, N, dtype=x.dtype, device=x.device)
    else:
        h = h0

    y = torch.empty(B, L, E, dtype=x.dtype, device=x.device)

    for t in range(L):
        # State update for every batch/channel pair.
        h = a_bar[:, t] * h + b_bar[:, t] * x[:, t].unsqueeze(-1)

        # c is shared across the E inner channels.
        y[:, t] = (c[:, t].unsqueeze(1) * h).sum(dim=-1)

    return y, h

# Step 14 - compare_constant_vs_selective_delta
def compare_constant_vs_selective_delta(x, a, b, c, delta_const, delta_sel):
    """Compare SSM scan outputs under a constant Delta versus a selective Delta.

    x: (batch, seq_len, d_inner)
    a: (d_inner, d_state) strictly negative continuous diagonal A
    b: (batch, seq_len, d_state)
    c: (batch, seq_len, d_state)
    delta_const: (batch, seq_len, d_inner) non-selective timestep
    delta_sel: (batch, seq_len, d_inner) input-dependent timestep

    Returns:
        y_const: (batch, seq_len, d_inner)
        y_sel: (batch, seq_len, d_inner)
    """
    a_bar_const = discretize_a_zoh(delta_const, a)
    b_bar_const = discretize_b_zoh(delta_const, a, b)

    a_bar_sel = discretize_a_zoh(delta_sel, a)
    b_bar_sel = discretize_b_zoh(delta_sel, a, b)

    y_const, _ = selective_scan(
        x, a_bar_const, b_bar_const, c
    )

    y_sel, _ = selective_scan(
        x, a_bar_sel, b_bar_sel, c
    )

    return y_const, y_sel

# Step 15 - gate_scan_output
def gate_scan_output(y, z):
    """Modulate the selective-scan output y by the parallel gate branch z."""
    return y * silu(z)

# Step 16 - out_proj
def out_proj(y, weight, bias=None):
    """Project gated scan output from d_inner back to d_model.

    y: (..., d_inner)
    weight: (d_model, d_inner)
    bias: (d_model,) or None
    Returns: (..., d_model)
    """
    output = y @ weight.transpose(-1, -2)

    if bias is not None:
        output = output + bias

    return output

# Step 17 - mamba_mixer
def mamba_mixer(u, params):
    """Run one full Mamba selective-SSM mixer on a token sequence.

    Args:
        u: (B, L, D) input sequence.
        params: dict of mixer weights. See the step description for keys.

    Returns:
        (B, L, D) mixer output.
    """
    # Input projection: split into SSM branch x and gate branch z.
    x, z = in_proj_split(
        u,
        params["in_proj_weight"],
        params.get("in_proj_bias"),
    )

    # Causal depthwise convolution followed by SiLU.
    x = causal_depthwise_conv1d(
        x,
        params["conv_weight"],
        params.get("conv_bias"),
    )
    x = silu(x)

    # Input-dependent timestep Delta.
    delta = compute_delta(
        x,
        params["dt_weight"],
        params.get("dt_bias"),
    )

    # Input-dependent SSM parameters B and C.
    B_ssm, C_ssm = project_bc(
        x,
        params["weight_b"],
        params["weight_c"],
    )

    # Log-parameterized continuous-time A.
    a = make_diagonal_a(params["log_a"])

    # Exact zero-order-hold discretization.
    a_bar = discretize_a_zoh(delta, a)
    b_bar = discretize_b_zoh(delta, a, B_ssm)

    # Sequential selective scan.
    y, _ = selective_scan(x, a_bar, b_bar, C_ssm)

    # Gated SSM output.
    y = gate_scan_output(y, z)

    # Project back to model width.
    y = out_proj(
        y,
        params["out_proj_weight"],
        params.get("out_proj_bias"),
    )

    return y

# Step 18 - mamba_block
def mamba_block(x, params):
    """Apply a pre-norm residual Mamba block to a token sequence.

    Args:
        x: (B, L, D) hidden sequence.
        params: dict with norm_weight (D,) plus every mamba_mixer key.

    Returns:
        (B, L, D) block output.
    """
    normalized = rms_norm(x, params["norm_weight"])
    mixer_output = mamba_mixer(normalized, params)

    return x + mixer_output

# Step 19 - run_mamba_lm_stack
def run_mamba_lm_stack(embeddings, params):
    """Run token embeddings through stacked Mamba residual blocks and a final RMSNorm.

    Args:
        embeddings: (B, L, D) token embeddings.
        params: dict with key `blocks` (list of per-block dicts for `mamba_block`)
            and key `norm_weight` of shape (D,) for the final RMSNorm (eps=1e-5).

    Returns:
        (B, L, D) hidden states after the stack and final RMSNorm.
    """
    hidden = embeddings

    for block_params in params["blocks"]:
        hidden = mamba_block(hidden, block_params)

    return rms_norm(hidden, params["norm_weight"])

# Step 20 - mamba_lm_forward
def mamba_lm_forward(token_ids, params):
    """Map token ids through embeddings, the Mamba stack, and an LM head.

    Args:
        token_ids: (B, L) integer tensor of token ids.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            blocks (list), and norm_weight (D,).

    Returns:
        (B, L, V) logits.
    """
    embeddings = torch.nn.functional.embedding(
        token_ids,
        params["embed_weight"],
    )

    hidden = run_mamba_lm_stack(embeddings, params)

    logits = hidden @ params["lm_head_weight"].transpose(-1, -2)

    return logits

# Step 21 - next_token_cross_entropy
def next_token_cross_entropy(logits, token_ids):
    """Compute the mean next-token cross-entropy from logits and token ids."""
    next_logits = logits[:, :-1, :]
    next_targets = token_ids[:, 1:]

    B, T_minus_1, V = next_logits.shape

    return F.cross_entropy(
        next_logits.reshape(B * T_minus_1, V),
        next_targets.reshape(B * T_minus_1),
        reduction="mean",
    )

# Step 22 - sgd_training_step
def sgd_training_step(token_ids, params, lr):
    """Run one vanilla SGD step of next-token prediction and return the loss.

    Args:
        token_ids: (B, L) integer tensor of token ids with L >= 2.
        params: dict with embed_weight (V, D), lm_head_weight (V, D),
            norm_weight (D,), and blocks (list of nested param dicts).
            Parameter tensors must have requires_grad=True and are updated in place.
        lr: vanilla SGD learning rate.

    Returns:
        Python float, the next-token cross-entropy from this step.
    """
    logits = mamba_lm_forward(token_ids, params)
    loss = next_token_cross_entropy(logits, token_ids)

    loss.backward()

    def update_params(obj):
        if isinstance(obj, torch.Tensor):
            if obj.requires_grad and obj.grad is not None:
                with torch.no_grad():
                    obj -= lr * obj.grad
                obj.grad = None

        elif isinstance(obj, dict):
            for value in obj.values():
                update_params(value)

        elif isinstance(obj, (list, tuple)):
            for value in obj:
                update_params(value)

    update_params(params)

    return loss.item()

# Step 23 - mamba_recurrent_step
def mamba_recurrent_step(token_ids, params, cache=None):
    """Consume one token and return next-token logits plus an updated SSM/conv cache."""

    # Accept token_ids with shape (B,) or (B, 1).
    if token_ids.ndim == 2:
        token_ids = token_ids.squeeze(1)

    # Embed the current token: (B,) -> (B, D).
    hidden = torch.nn.functional.embedding(
        token_ids,
        params["embed_weight"],
    )

    num_layers = len(params["blocks"])

    # Initialize recurrent states if this is the first token.
    if cache is None:
        conv_states = [None] * num_layers
        ssm_states = [None] * num_layers
    else:
        conv_states = cache["conv_states"]
        ssm_states = cache["ssm_states"]

    new_conv_states = []
    new_ssm_states = []

    for layer_idx, block_params in enumerate(params["blocks"]):
        # ------------------------------------------------------------
        # Pre-norm residual block
        # ------------------------------------------------------------
        residual = hidden

        hidden = rms_norm(
            hidden.unsqueeze(1),
            block_params["norm_weight"],
        ).squeeze(1)

        # ------------------------------------------------------------
        # Input projection: x branch + gate branch z
        # ------------------------------------------------------------
        x, z = in_proj_split(
            hidden.unsqueeze(1),
            block_params["in_proj_weight"],
            block_params.get("in_proj_bias"),
        )

        x = x.squeeze(1)
        z = z.squeeze(1)

        B, E = x.shape
        K = block_params["conv_weight"].shape[1]

        # ------------------------------------------------------------
        # Causal depthwise convolution
        #
        # The recurrent cache stores the previous K-1 RAW x values.
        # The current convolution window is:
        #
        # [x_{t-K+1}, ..., x_{t-1}, x_t]
        # ------------------------------------------------------------
        previous_conv = conv_states[layer_idx]

        if previous_conv is None:
            previous_conv = torch.zeros(
                B,
                K - 1,
                E,
                dtype=x.dtype,
                device=x.device,
            )

        conv_window = torch.cat(
            [previous_conv, x.unsqueeze(1)],
            dim=1,
        )

        # conv_window: (B, K, E)
        # conv_weight: (E, K)
        #
        # Transpose weight to (K, E) so each channel gets its own
        # length-K kernel.
        conv_weight = block_params["conv_weight"].transpose(0, 1)

        x_conv = (
            conv_window * conv_weight.unsqueeze(0)
        ).sum(dim=1)

        if block_params.get("conv_bias") is not None:
            x_conv = x_conv + block_params["conv_bias"]

        # SiLU after convolution.
        x_conv = silu(x_conv)

        # Keep the newest K-1 RAW convolution inputs.
        if K > 1:
            new_conv_state = conv_window[:, -(K - 1):, :]
        else:
            new_conv_state = conv_window[:, :0, :]

        new_conv_states.append(new_conv_state)

        # ------------------------------------------------------------
        # Selective SSM parameters
        # ------------------------------------------------------------
        delta = compute_delta(
            x_conv.unsqueeze(1),
            block_params["dt_weight"],
            block_params.get("dt_bias"),
        )

        B_ssm, C_ssm = project_bc(
            x_conv.unsqueeze(1),
            block_params["weight_b"],
            block_params["weight_c"],
        )

        # Continuous-time A.
        a = make_diagonal_a(block_params["log_a"])

        # Exact ZOH discretization.
        a_bar = discretize_a_zoh(delta, a)
        b_bar = discretize_b_zoh(delta, a, B_ssm)

        # ------------------------------------------------------------
        # Recurrent SSM state
        # ------------------------------------------------------------
        if ssm_states[layer_idx] is None:
            N = a.shape[-1]

            h = torch.zeros(
                B,
                E,
                N,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            h = ssm_states[layer_idx]

        # One timestep of the selective SSM recurrence.
        h = (
            a_bar[:, 0] * h
            + b_bar[:, 0] * x_conv.unsqueeze(-1)
        )

        new_ssm_states.append(h)

        # ------------------------------------------------------------
        # SSM readout
        # ------------------------------------------------------------
        y = (
            C_ssm[:, 0].unsqueeze(1) * h
        ).sum(dim=-1)

        # ------------------------------------------------------------
        # Gating
        # ------------------------------------------------------------
        y = gate_scan_output(
            y.unsqueeze(1),
            z.unsqueeze(1),
        ).squeeze(1)

        # ------------------------------------------------------------
        # Output projection + residual
        # ------------------------------------------------------------
        y = out_proj(
            y,
            block_params["out_proj_weight"],
            block_params.get("out_proj_bias"),
        )

        hidden = residual + y

    # ------------------------------------------------------------
    # Final RMSNorm
    # ------------------------------------------------------------
    hidden = rms_norm(
        hidden.unsqueeze(1),
        params["norm_weight"],
    ).squeeze(1)

    # ------------------------------------------------------------
    # Bias-free LM head
    # ------------------------------------------------------------
    logits = (
        hidden
        @ params["lm_head_weight"].transpose(-1, -2)
    )

    new_cache = {
        "conv_states": new_conv_states,
        "ssm_states": new_ssm_states,
    }

    return logits, new_cache

# Step 24 - greedy_generate
def greedy_generate(prompt_ids, params, max_new_tokens):
    """Greedily generate new token ids from a prompt using a carried SSM cache."""
    if max_new_tokens == 0:
        return prompt_ids.clone()

    generated = prompt_ids.clone()
    cache = None

    # Consume the entire prompt to build the recurrent cache.
    logits = None

    for token in prompt_ids:
        logits, cache = mamba_recurrent_step(
            token.unsqueeze(0),
            params,
            cache,
        )

    # Generate one token at a time using the carried cache.
    for _ in range(max_new_tokens):
        # argmax chooses the lowest index automatically when logits tie.
        next_token = torch.argmax(logits, dim=-1)

        generated = torch.cat(
            [generated, next_token.to(generated.device).reshape(1)]
        )

        # Feed the newly generated token back into the recurrent model.
        logits, cache = mamba_recurrent_step(
            next_token,
            params,
            cache,
        )

    return generated

# Step 25 - train_tiny_mamba_and_generate
import math

def train_tiny_mamba_and_generate(
    corpus,
    n_steps,
    lr,
    prompt,
    max_new_tokens,
    d_model=16,
    n_layers=2,
    d_state=4,
    d_inner=32,
    conv_kernel=3,
    seed=0,
):
    """Train a tiny character-level Mamba LM on corpus and greedily generate from prompt."""

    # ------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------
    vocab = sorted(set(corpus))
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    id_to_char = {i: ch for i, ch in enumerate(vocab)}

    V = len(vocab)

    corpus_ids = torch.tensor(
        [[char_to_id[ch] for ch in corpus]],
        dtype=torch.long,
    )

    prompt_ids = torch.tensor(
        [char_to_id[ch] for ch in prompt],
        dtype=torch.long,
    )

    # ------------------------------------------------------------
    # Reproducible parameter initialization
    # ------------------------------------------------------------
    torch.manual_seed(seed)

    def normal_weight(*shape):
        return torch.randn(
            *shape,
            dtype=torch.float32,
        ) * 0.02

    def parameter(tensor):
        return tensor.detach().clone().requires_grad_(True)

    def make_weight(*shape):
        return parameter(normal_weight(*shape))

    def make_bias(size):
        return parameter(torch.zeros(size, dtype=torch.float32))

    def make_norm(size):
        return parameter(torch.ones(size, dtype=torch.float32))

    # ------------------------------------------------------------
    # Top-level parameters
    # ------------------------------------------------------------
    params = {
        "embed_weight": make_weight(V, d_model),
        "lm_head_weight": make_weight(V, d_model),
        "norm_weight": make_norm(d_model),
        "blocks": [],
    }

    # ------------------------------------------------------------
    # Mamba blocks
    # ------------------------------------------------------------
    for _ in range(n_layers):
        # log_a[i, n] = log(n + 1)
        log_a = torch.log(
            torch.arange(
                1,
                d_state + 1,
                dtype=torch.float32,
            )
        ).unsqueeze(0).expand(d_inner, -1).clone()

        block = {
            "norm_weight": make_norm(d_model),

            "in_proj_weight": make_weight(
                2 * d_inner,
                d_model,
            ),
            "in_proj_bias": make_bias(2 * d_inner),

            "conv_weight": make_weight(
                d_inner,
                conv_kernel,
            ),
            "conv_bias": make_bias(d_inner),

            "dt_weight": make_weight(
                d_inner,
                d_inner,
            ),
            "dt_bias": make_bias(d_inner),

            "weight_b": make_weight(
                d_state,
                d_inner,
            ),
            "weight_c": make_weight(
                d_state,
                d_inner,
            ),

            "log_a": parameter(log_a),

            "out_proj_weight": make_weight(
                d_model,
                d_inner,
            ),
            "out_proj_bias": make_bias(d_model),
        }

        params["blocks"].append(block)

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    losses = []

    for _ in range(n_steps):
        loss = sgd_training_step(
            corpus_ids,
            params,
            lr,
        )
        losses.append(loss)

    # ------------------------------------------------------------
    # Greedy recurrent generation
    # ------------------------------------------------------------
    generated_ids = greedy_generate(
        prompt_ids,
        params,
        max_new_tokens,
    )

    # ------------------------------------------------------------
    # Decode generated token ids
    # ------------------------------------------------------------
    text = "".join(
        id_to_char[int(token_id)]
        for token_id in generated_ids
    )

    return text, losses

