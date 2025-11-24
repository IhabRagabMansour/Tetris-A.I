# Weights & Biases (wandb) Integration Guide

## 🎯 Overview

This project now includes **Weights & Biases** integration to track and compare experiments between **CNN** and **Linear** Tetris AI architectures.

## 📦 Installation

1. Install wandb:
```bash
pip install wandb
```

2. Login to wandb (first time only):
```bash
wandb login
```
This will open your browser to get an API key. Copy and paste it into the terminal.

## 🚀 Quick Start

### Option 1: Using the Experiment Runner (Recommended)

```bash
# Train CNN model with wandb
python run_experiments.py --model cnn --wandb --games 10000

# Train Linear model with wandb
python run_experiments.py --model linear --wandb --games 10000

# Run both experiments sequentially
python run_experiments.py --model both --wandb --games 10000
```

### Option 2: Using train.py Directly

Edit `train.py` at the bottom (`if __name__=='__main__':`):

```python
# Train CNN with wandb
run_game(
    games=10000,
    use_wandb=True,
    experiment_name="cnn-tetris-v1",
    architecture="CNN"
)
```

Then run:
```bash
python train.py
```

## 📊 Metrics Tracked

### Per-Game Metrics (logged every game):
- `game_number` - Current game number
- `game_lines` - Lines cleared in this game
- `game_score` - Score achieved in this game
- `epsilon` - Current exploration rate
- `learning_rate` - Current learning rate

### Aggregate Metrics (logged every 100 games):
- `avg_lines_overall` - Average lines across all games
- `avg_lines_recent_100` - Average lines in last 100 games
- `total_lines` - Cumulative lines cleared
- `tetris_clears` - Number of 4-line clears
- `early_failure_rate` - % of games ending with <20 lines
- `avg_score` - Average score across all games
- `avg_q_value` - Average Q-value (model confidence)

### Final Summary:
- `final_total_lines` - Total lines across all games
- `final_total_tetris_clears` - Total Tetris clears
- `training_time_seconds` - Total training time
- `training_time_hours` - Total training time in hours

## 🔬 Experiment Configurations Tracked

All hyperparameters are automatically logged:
- Architecture (CNN vs Linear)
- Max Memory size
- Batch size
- Learning rate (initial and final)
- Epsilon decay strategy
- Gamma (discount factor)
- Hidden layer sizes
- Reward function weights

## 📈 Viewing Results

After starting training with `--wandb`, you'll see:
```
Weights & Biases initialized: cnn-tetris-v1
```

Visit https://wandb.ai/ to see:
- **Real-time graphs** of all metrics
- **Side-by-side comparisons** of CNN vs Linear
- **Interactive dashboards**
- **Hyperparameter tracking**

## 🎓 Comparing CNN vs Linear

### Step 1: Train CNN Model
```bash
python run_experiments.py --model cnn --wandb --games 10000
```

### Step 2: Switch to Linear Architecture

**Edit `agent.py` line 4:**
```python
# Change from:
from model import QTrainer, CNN_QNet

# To:
from model import QTrainer, Linear_QNet
```

**Edit `agent.py` lines 31-32:**
```python
# Change from:
self.model1 = CNN_QNet(output_size=self.ACTIONS).to(self.device)
self.model2 = CNN_QNet(output_size=self.ACTIONS).to(self.device)

# To:
self.model1 = Linear_QNet(self.STATES, self.HIDDEN_SIZES, self.ACTIONS).to(self.device)
self.model2 = Linear_QNet(self.STATES, self.HIDDEN_SIZES, self.ACTIONS).to(self.device)
```

### Step 3: Train Linear Model
```bash
python run_experiments.py --model linear --wandb --games 10000
```

### Step 4: Compare Results

Visit your wandb project: https://wandb.ai/YOUR_USERNAME/tetris-ai-comparison

You'll see both runs side-by-side with comparison graphs!

## 📊 Key Comparisons to Look For

1. **Average Lines Over Time**
   - Which architecture learns faster?
   - Which reaches higher peak performance?

2. **Early Failure Rate**
   - Which is more stable?
   - Which produces fewer catastrophic failures?

3. **Training Time**
   - How much longer does CNN take?
   - Is the performance gain worth the compute cost?

4. **Q-Value Confidence**
   - Which model is more confident in its decisions?

5. **Tetris Clears**
   - Which model shows better strategic depth?

## 🎯 Expected Results

### CNN Advantages:
- ✅ Higher ceiling for performance (potential for 50-80+ avg lines)
- ✅ Can discover spatial patterns humans might miss
- ✅ No manual feature engineering required

### Linear Advantages:
- ✅ Faster training (fewer parameters)
- ✅ More interpretable (can see which features matter)
- ✅ Lower memory requirements

## 🛠️ Advanced Usage

### Resume Training with wandb
```python
run_game(
    games=15000,
    resume_from="model/trained_model_10.pth",
    use_wandb=True,
    experiment_name="cnn-tetris-v1",  # Same name continues the run
    architecture="CNN"
)
```

### Custom Project Name
```python
run_game(
    games=10000,
    use_wandb=True,
    experiment_name="my-experiment",
    architecture="CNN",
    project_name="my-custom-project"
)
```

### Disable wandb Temporarily
```python
run_game(games=10000, use_wandb=False)
```

## 📝 Creating Reports

In wandb web interface:
1. Go to your project
2. Click "Reports" → "Create Report"
3. Add comparison graphs
4. Add commentary about findings
5. Share with your professor!

## 🐛 Troubleshooting

**"wandb not found"**
```bash
pip install wandb
```

**"wandb login required"**
```bash
wandb login
```

**Runs not showing up**
- Check you're logged in: `wandb whoami`
- Check internet connection
- Verify project name is correct

**Want to run offline**
```bash
wandb offline
python train.py
```

## 📚 Resources

- wandb Documentation: https://docs.wandb.ai/
- wandb Python API: https://docs.wandb.ai/ref/python
- Example Reports: https://wandb.ai/site/reports

## ✅ Checklist for Your Comparison Experiment

- [ ] Install wandb: `pip install wandb`
- [ ] Login: `wandb login`
- [ ] Train CNN: `python run_experiments.py --model cnn --wandb`
- [ ] Switch to Linear_QNet in agent.py
- [ ] Train Linear: `python run_experiments.py --model linear --wandb`
- [ ] View results at wandb.ai
- [ ] Create comparison report
- [ ] Share findings with professor!

Good luck with your experiments! 🎓🚀
