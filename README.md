# 2048 Game LLM Training with GRPO

## Overview
This project implements a Generative Reinforcement Learning from Preference Optimization (GRPO) approach to train Large Language Models (LLMs) for playing the 2048 game. The training pipeline includes dataset generation, model fine-tuning, and comprehensive reward engineering.

## Features
- **Dataset Generation**: Creates diverse 2048 game states with difficulty classifications
- **Multi-Modal Support**: Handles both text-only and visual (VLM) training formats
- **Advanced Reward Functions**:
  - Tag structure validation
  - Move validity checking
  - Density-based rewards
  - Survival rewards
  - High-tile achievement rewards

## Technical Details
- **Base Model**: Qwen 2.5 7B Instruct
- **Training Method**: GRPO (Generative Reinforcement Learning from Preference Optimization)
- **LoRA Parameters**:
  - Rank: 16
  - Target Modules: QKVO and MLP
  - Learning Rate: 4e-5

## Dataset Characteristics
- Training Size: 8000 samples
- Test Size: 50 samples
- 5 difficulty levels with distribution [1,2,3,4,5]
- Board states generated with varying complexity

## Training Configuration
- Batch Size: 1
- Gradient Accumulation Steps: 4
- Max Steps: 800
- Evaluation Strategy: Steps-based
- Weight Decay: 0.01
- GPU Memory Utilization: 0.8

## Requirements
- unsloth
- trl==0.15.2
- torch
- transformers
- wandb (for logging)
- numpy
- pandas
- matplotlib

## Usage
1. Configure environment variables and dependencies
2. Generate dataset using the provided creator class
3. Initialize model and training configuration
4. Run training with GRPO trainer
5. Save and evaluate model checkpoints

## Model Outputs
The training produces:
- LoRA weights for efficient fine-tuning
- Merged model weights for inference
- Training logs and metrics
- Performance visualizations

## Monitoring
- WandB integration for training metrics
- SQLite database for response logging
- Comprehensive error tracking
- Performance statistics collection

## License
Apache 2.0

## Citation
@misc{dalal2025teaching2048,
author = {Dalal, Hrishbh},
title = {{Agent 2048: Forging Strategic Gameplay in an AI Through Data, Rewards, and RL
}},
year = {2025},
month = {April},
day = {1},
url = {[https://yourwebsite.com/blog/ai-agent-plays-2048](https://hrishbh.com/agent-2048-forging-strategic-gameplay-in-an-ai-through-data-rewards-and-rl/)},
note = {Accessed on April 1, 2025}
}

## Acknowledgments
- Qwen team for the base model
- Unsloth team for optimization tools
- WandB for experiment tracking
