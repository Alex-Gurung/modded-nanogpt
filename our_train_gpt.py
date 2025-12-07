"""
Agent-facing training entrypoint for modded-nanogpt.

The heavy-weight training loop, kernels, data pipeline, and distributed setup
live in agent_core/base_train.py. This file only wires up hook functions the
automated agent is allowed to edit. To try new ideas, tweak the hook functions
below or mutate BASE_ARGS; the core loop stays frozen in the utilities.
"""

from __future__ import annotations

from agent_core import base_train
from agent_core.base_train import AgentHooks, Hyperparameters, RuntimeState

# ---------------------------------------------------------------------------
# Agent-editable surface
# ---------------------------------------------------------------------------

# Start from these defaults when running; edit freely to explore ideas.
BASE_ARGS = Hyperparameters()


def mutate_hparams(args: Hyperparameters) -> Hyperparameters:
    """
    Return a possibly modified Hyperparameters object.

    The baseline simply returns the defaults. Agents can adjust learning rates,
    schedules, batch sizes, or anything else exposed on Hyperparameters here.
    """
    return args


def build_model(args: Hyperparameters, runtime: RuntimeState):
    """
    Optional custom model builder.

    Stick with the baseline GPT by default. Override to experiment with new
    architectures, extra layers, or altered initialization while keeping the
    training loop unchanged.
    """
    return base_train._build_default_model(args, runtime)


def build_optimizers(model, args: Hyperparameters, runtime: RuntimeState):
    """
    Optional optimizer factory.

    Use the standard DistAdam + NorMuon setup by default. Override to try new
    optimizers, parameter grouping strategies, or lr/weight-decay settings.
    """
    return base_train._build_default_optimizers(model, args, runtime)


def step_optimizers(step: int, optimizers, model, args: Hyperparameters, runtime: RuntimeState):
    """
    Optional optimizer stepping policy.

    The default mirrors train_gpt.py: schedule LR/momentum and alternate between
    Muon-only and all-params steps. Override to change scheduling logic without
    touching the core training loop.
    """
    return base_train.step_optimizers(step, optimizers, model)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main():
    hooks = AgentHooks(
        mutate_hparams=mutate_hparams,
        build_model=build_model,
        build_optimizers=build_optimizers,
        step_optimizers=step_optimizers,
    )
    base_train.run_training(agent_hooks=hooks, base_args=BASE_ARGS)


if __name__ == "__main__":
    main()
