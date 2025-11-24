"""
Experiment Runner for Comparing CNN vs Linear Tetris AI

This script makes it easy to run comparison experiments with Weights & Biases.

Usage:
    python run_experiments.py --model cnn --wandb
    python run_experiments.py --model linear --wandb
    python run_experiments.py --model both --wandb  # Run both sequentially
"""

import argparse
from train import run_game

def run_cnn_experiment(games=10000, use_wandb=True):
    """Run CNN-based Tetris AI experiment"""
    print("="*60)
    print("EXPERIMENT: CNN Architecture (Raw 20x10 Board Input)")
    print("="*60)

    run_game(
        games=games,
        use_wandb=use_wandb,
        experiment_name="cnn-tetris-raw-board",
        architecture="CNN",
        project_name="tetris-ai-comparison"
    )

def run_linear_experiment(games=10000, use_wandb=True):
    """Run Linear-based Tetris AI experiment"""
    print("="*60)
    print("EXPERIMENT: Linear Architecture (6 Engineered Features)")
    print("="*60)
    print("\n⚠️  WARNING: Make sure you've switched to Linear_QNet in agent.py!")
    print("    Change line: from model import QTrainer, CNN_QNet")
    print("    To:          from model import QTrainer, Linear_QNet")
    print("    And update model1/model2 initialization in agent.py\n")

    response = input("Have you switched to Linear_QNet? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborting. Please switch to Linear_QNet first.")
        return

    run_game(
        games=games,
        use_wandb=use_wandb,
        experiment_name="linear-tetris-engineered-features",
        architecture="Linear",
        project_name="tetris-ai-comparison"
    )

def main():
    parser = argparse.ArgumentParser(description='Run Tetris AI comparison experiments')
    parser.add_argument('--model', type=str, choices=['cnn', 'linear', 'both'],
                        default='cnn', help='Which model to train')
    parser.add_argument('--games', type=int, default=10000,
                        help='Number of games to train')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    args = parser.parse_args()

    use_wandb = args.wandb and not args.no_wandb

    if use_wandb:
        print("\n✅ Weights & Biases logging enabled")
        print("📊 View results at: https://wandb.ai/\n")
    else:
        print("\n❌ Weights & Biases logging disabled")
        print("💡 Add --wandb flag to enable experiment tracking\n")

    if args.model == 'cnn':
        run_cnn_experiment(args.games, use_wandb)
    elif args.model == 'linear':
        run_linear_experiment(args.games, use_wandb)
    elif args.model == 'both':
        print("\n🔬 Running BOTH experiments sequentially...\n")
        run_cnn_experiment(args.games, use_wandb)
        print("\n" + "="*60)
        print("CNN Experiment Complete! Starting Linear Experiment...")
        print("="*60 + "\n")
        run_linear_experiment(args.games, use_wandb)

    print("\n✅ All experiments completed!")
    if use_wandb:
        print("📊 View comparison at: https://wandb.ai/")

if __name__ == '__main__':
    main()
