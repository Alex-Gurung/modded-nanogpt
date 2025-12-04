# Bio-inspired training experiments for modded-nanogpt

This document describes the experimental changes added on top of `train_gpt.py`.  
All changes are optional and controlled via new fields in the `Hyperparameters` dataclass.

The goals:

1. Give the training loop a **spaced-repetition-like mechanism** for hard batches, without changing the dataset format.
2. Allow the loss to **focus gradient budget on non-trivially-easy tokens**, inspired by desirable difficulty.
3. Add a **light form of structural sparsity** at the level of rows of MLP/attention matrices:
   - frequently useful rows get more updates,
   - consistently unimportant rows can be reset and repurposed.

None of these change the model architecture or the optimizer internals. They operate purely at:
- Data sampling (replay vs fresh batch),
- Loss aggregation (per-token weighting),
- Gradient post-processing (gating / reset).

---

## 1. New hyperparameters

All new fields live in `Hyperparameters` inside `train_gpt.py`. They are grouped by function.

### 1.1 Difficulty-weighted loss (inside `GPT.forward`)

*Fields:*

- `enable_difficulty_weighting: bool`  
  If `True`, the model downweights trivially easy tokens when computing the cross-entropy loss.

- `difficulty_easy_z: float`  
  Threshold on a token’s **loss z-score**. Tokens whose loss is far below the running mean (e.g. z < -1) are treated as “easy”.

- `difficulty_easy_weight: float`  
  Weight assigned to easy tokens. For example, `0.3` means each easy-token loss contributes 30% as much as a normal token.

- `difficulty_ema_beta: float`  
  Decay factor for the running mean/variance of per-token loss. Higher values (e.g. 0.99) make the stats smoother.

*Mechanics:*

- `GPT.forward` now maintains two scalar buffers:
  - `loss_mean_ema`
  - `loss_var_ema`
- Each training call:
  1. Computes per-token cross-entropy losses.
  2. Updates the running mean and variance with an EMA.
  3. Computes a z-score for each token loss.
  4. Leaves hard/normal tokens unchanged, but multiplies “easy” ones (z below `difficulty_easy_z`) by `difficulty_easy_weight`.
- The loss is the **sum** of the weighted token losses (preserving original scaling with the training schedule).

*Intuition:*

This is a gentle way of:

- Preserving all information (no token is ignored),
- But spending less gradient on tokens that are clearly mastered (analogous to an SRS system not focusing on flashcards you always get right).

---

### 1.2 Hard batch replay (SRS-ish behavior over batches)

This keeps a small buffer of “hard” batches and occasionally replays them, spaced out over time.

*Fields:*

- `enable_hard_replay: bool`  
  Master switch. When `False`, the buffer is never used.

- `hard_buffer_size: int`  
  Maximum number of batches stored in the replay buffer.

- `hard_base_interval: int`  
  Initial spacing (in **training steps**) before a stored batch is eligible for replay again.

- `hard_interval_growth: float`  
  Multiplier for the spacing after each review. E.g. 2.0 means intervals double each time a batch is revisited.

- `hard_min_loss_scale: float`  
  Batches must have loss at least `min_loss_scale * running_loss_ema` to be admitted. This avoids filling the buffer with only moderately-difficult examples.

- `hard_loss_ema_beta: float`  
  EMA decay for the running “typical” train loss used to normalize difficulty.

- `hard_replay_fraction: float`  
  Fraction of **gradient accumulation microsteps** that are allowed to use a replay batch.  
  For example, `0.25` means that on average one out of four microsteps will use replay (if due batches exist).

*Mechanics:*

- A `HardReplayBuffer` is instantiated after model creation, using `HardReplayConfig`.
- Each fresh batch:
  1. Computes the loss.
  2. Updates the buffer’s loss EMA.
  3. If the loss is sufficiently high (controlled by `min_loss_scale`), the batch is copied to CPU and stored with:
     - its current loss,
     - the step at which it’s next due for replay,
     - a count of how many times it has been seen.
- On each microstep:
  - With probability `hard_replay_fraction`, the trainer tries to sample a replay batch that is **due at the current step**.
  - If none are due, or replay is disabled, it falls back to fresh data.
- When a replay batch is used:
  - Its new loss is measured.
  - It is rescheduled with a larger interval between reviews, proportional to how many times it has been seen already.

*Intuition:*

This is a practical, local approximation to spaced repetition:

- Batches that repeatedly yield high loss remain in circulation longer,
- Batches that become easier over time automatically fall below the admission threshold and stop crowding the buffer.

Because this acts on top of the existing streaming data loader, it doesn’t require multiple full epochs or structural changes to the `.bin` file layout.

---

### 1.3 Neuron gating and tag-and-reset

This provides a light form of structural sparsity and continual learning at the level of **rows of weight matrices**.

*Fields:*

- `enable_neuron_gating: bool`  
  If `True`, low-importance rows in selected parameter matrices have their gradients zeroed before each optimizer step.

- `neuron_gating_topk_fraction: float`  
  Fraction of rows to keep. For example:
  - `0.5` → only the top 50% by EMA grad norm receive updates,
  - `1.0` → gating disabled in practice (all rows kept).

- `enable_neuron_reset: bool`  
  If `True`, periodically reinitializes a fraction of consistently low-EMA rows, recycling “dead” neurons.

- `neuron_reset_interval: int`  
  Steps between reset sweeps.

- `neuron_reset_quantile: float`  
  Bottom quantile of rows to reset. E.g. `0.05` resets the lowest 5% of rows by EMA grad norm.

- `neuron_grad_ema_decay: float`  
  EMA decay factor for per-row gradient norms.

*Mechanics:*

- After compiling the model, the script calls `init_neuron_stats(model, device, cfg)`:
  - It looks through all parameters,
  - Selects those with `param.label` in `("mlp", "attn")` and `ndim >= 2`,
  - Allocates a vector of zeros of length `param.shape[0]` to track per-row grad EMAs.
- After each **full training step** (after all `grad_accum_steps` backward calls, but before stepping the optimizers), the script calls:
  ```python
  apply_neuron_gating_and_reset(neuron_stats, step, neuron_gating_cfg)

