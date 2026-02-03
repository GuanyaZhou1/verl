# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It implements the HybridFlow architecture for RLHF with support for PPO, GRPO, and other RL algorithms.

## Common Commands

### Installation

```bash
# Development installation with test dependencies
pip install -e .[test,vllm]   # for vLLM backend
pip install -e .[test,sglang] # for SGLang backend
```

### Running Training

Training uses Hydra configuration. The main entry point is `verl.trainer.main_ppo`:

```bash
# Run GRPO training example
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=path/to/train.parquet \
    data.val_files=path/to/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    trainer.n_gpus_per_node=8

# Run with shell script (contains full config)
bash examples/grpo_trainer/run_qwen3-8b.sh
```

Override any config parameter via command line using Hydra syntax (dot notation).

### Linting and Formatting

```bash
pip install pre-commit
pre-commit install
pre-commit run                    # staged changes only
pre-commit run --all-files        # entire repo
pre-commit run --all-files ruff   # run specific hook
```

### Running Tests

Tests are organized under `tests/`:
- `tests/trainer/`, `tests/models/`, etc. - unit tests by namespace
- `tests/special_distributed/` - multi-GPU tests
- `tests/special_e2e/` - end-to-end training tests
- Files ending in `_on_cpu.py` run on CPU; others require GPU

```bash
# Run a specific test file
pytest tests/trainer/test_something.py

# Run tests for a module
pytest tests/trainer/
```

### Building Documentation

```bash
cd docs
pip install -r requirements-docs.txt
make clean && make html
python -m http.server -d _build/html/  # preview at localhost:8000
```

## Architecture Overview

### Core Components

```
verl/
├── trainer/           # Training orchestration
│   ├── main_ppo.py    # Main entry point (Hydra-based)
│   ├── ppo/           # PPO/GRPO trainer implementations
│   │   ├── ray_trainer.py  # RayPPOTrainer - main training loop
│   │   └── core_algos.py   # Advantage estimators, policy losses
│   └── config/        # Hydra YAML configurations
├── workers/           # Distributed worker implementations
│   ├── rollout/       # Text generation (vLLM, SGLang, TensorRT-LLM, HF)
│   ├── actor/         # Policy training (dp_actor, megatron_actor)
│   ├── critic/        # Value function training
│   ├── reward_manager/# Reward computation backends
│   └── engine/        # Training engines (FSDP, Megatron)
├── single_controller/ # Ray-based distributed coordination
│   └── ray/           # Worker groups and resource pools
├── models/            # Model implementations and configs
├── protocol.py        # DataProto - unified data structure
└── experimental/      # Research features (async training, VLA, etc.)
```

### Training Flow

1. **RayPPOTrainer** (`verl/trainer/ppo/ray_trainer.py`) orchestrates training:
   - Creates Ray resource pools for Actor, Critic, Rollout, RewardModel workers
   - Manages training loop: rollout → reward → advantage → policy update

2. **Workers** are distributed across GPUs:
   - **Rollout Workers**: Generate text using vLLM/SGLang
   - **Actor Workers**: Compute log_probs, update policy (FSDP/Megatron)
   - **Critic Workers**: Compute value estimates (PPO only)
   - **Reward Manager**: Compute rewards (rule-based or model-based)

3. **DataProto** (`verl/protocol.py`) is the unified data structure passed between components, built on tensordict.

### Key Configuration Files

- `verl/trainer/config/ppo_trainer.yaml` - Main PPO/GRPO config
- `verl/trainer/config/ppo_megatron_trainer.yaml` - Megatron backend config
- `verl/trainer/config/sft_trainer.yaml` - Supervised fine-tuning config

### Algorithm Selection

Set `algorithm.adv_estimator` to choose the algorithm:
- `gae` - PPO with Generalized Advantage Estimation (requires critic)
- `grpo` - Group Relative Policy Optimization (critic-less)
- `reinforce_plus_plus` - REINFORCE++
- `rloo` - RLOO
- `remax` - ReMax

### Training Backends

- **FSDP/FSDP2**: Set `actor_rollout_ref.actor.strategy=fsdp` or `fsdp2`
- **Megatron-LM**: Use `ppo_megatron_trainer.yaml` config

### Rollout Engines

- **vLLM**: `actor_rollout_ref.rollout.name=vllm`
- **SGLang**: `actor_rollout_ref.rollout.name=sglang`
- **TensorRT-LLM**: `actor_rollout_ref.rollout.name=trtllm`

## Important Patterns

### Hydra Config Overrides

Configs use composable defaults. Override via CLI:
```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=model_name \
    actor_rollout_ref.rollout.n=5 \
    data.train_batch_size=1024
```

### Worker Roles

Defined in `verl/trainer/ppo/utils.py`:
- `Role.Actor` - Policy training
- `Role.Rollout` - Text generation
- `Role.Critic` - Value function
- `Role.RefPolicy` - Reference policy for KL penalty
- `Role.ActorRolloutRef` - Combined actor + rollout + ref (common colocated setup)

### Reward Functions

Custom reward functions are specified in config:
```yaml
custom_reward_function:
  path: path/to/reward.py
  name: compute_score
```

The function receives batch data and returns token-level or sequence-level scores.
