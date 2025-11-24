from tetris import Tetris
from agent import Agent
#from plot import plot
import cProfile
import pstats
import wandb
import time

LR = 0.01
STATES = 6
HIDDEN_SIZES = [32,32,32]
INPUT_CHANNELS = 1  # For CNN
ACTIONS = 1
MAX_MEMORY = 100000  # Increased from 30000 to prevent good experiences from being overwritten
BATCH_SIZE = 128
EPOCHS = 2

class Training_Simulation:
    def __init__(self, genome, i, generation, total_games, SLOW_DROP=True, resume_from=None, use_wandb=False, architecture="CNN"):
        self.generation = generation
        self.i = i
        self.architecture = architecture
        self.tetris = Tetris(i=i, SLOW_DROP=SLOW_DROP, architecture=architecture)
        self.weight = genome
        # self.data = [MAX_MEMORY, STATES, HIDDEN_SIZES, ACTIONS, BATCH_SIZE, LR, EPOCHS, total_games]
        self.data = [MAX_MEMORY, STATES, HIDDEN_SIZES, ACTIONS, BATCH_SIZE, LR, EPOCHS, total_games]
        self.agent = Agent(self.data, architecture=architecture)
        self.use_wandb = use_wandb

        # Load checkpoint if resuming
        self.start_game = 1
        if resume_from:
            loaded_game = self.agent.load_model(resume_from)
            self.start_game = loaded_game + 1
            self.tetris.games = loaded_game
            print(f"Resuming training from game {self.start_game}")

    def calculate_rewards(self,best_state):
        total_heights, bumpiness, lines_removed, holes, y_pos, pillar = best_state
        calc_reward = 0

        # Define when the board is "half-full"
        board_half_full = total_heights >= 110 or (total_heights >= 90 and bumpiness >= 10)

        if total_heights >= 140 or (total_heights >= 110 and bumpiness >= 12):
            hole_penalty = -2.743561101942274  # Reduced penalty when board is high
        elif total_heights >= 90 or (total_heights >= 70 and bumpiness >= 9):
            hole_penalty = -4.743561101942274
        else:
            hole_penalty = self.weight['holes']

        # Discourage Placing High When Board is Low
        if total_heights <= 40:  # Board is mostly empty
            high_placement_penalty = (10 - y_pos) * 2  # Stronger penalty
        elif total_heights <= 100:  # Board is partially filled
            high_placement_penalty = (10 - y_pos)  # Moderate penalty
        else:
            high_placement_penalty = 0  # No penalty when the board is high

        # If the piece is placed in the upper 40% of the board
        if y_pos >= 12:
            calc_reward -= high_placement_penalty

        # Game over penalty
        if self.tetris.game.finished:
            calc_reward -= self.weight['game_over']  # Severe punishment for game over

        # Base survival incentive
        calc_reward += self.weight['survival_instinct']

        # Low piece placement reward (encourage low stacking)
        if y_pos >= 9:
            calc_reward += self.weight['y_pos_reward']  # Reward for stacking high when appropriate
        else:
            calc_reward -= (10 - y_pos) * 0.2 - self.weight['y_pos_punish']  # Gradual penalty for high stacking

        # Penalty for total board height
        calc_reward += self.weight['total_height'] * total_heights  # Encourage keeping the board low

        # Line clear reward (scaled)
        calc_reward += (2 ** lines_removed) * self.weight['lines_removed']  # Scaled reward for big clears
        if lines_removed == 4:
            calc_reward += 5000

        # Penalty for holes
        calc_reward += hole_penalty * holes

        # Penalty for bumpiness (prefer smooth board)
        calc_reward += self.weight['bumpiness'] * bumpiness

        # Pillar penalty
        pillar_penalty = 0
        if holes > 0 or board_half_full:
            pillar_penalty = self.weight['pillar']
        calc_reward += pillar_penalty

        return calc_reward

    def run_simulation(self,n):
        tetris = self.tetris
        agent = self.agent
        score = lines = not_trained = 0
        tetris_clears = 0
        count = (self.start_game - 1) // 500  # Calculate checkpoint counter based on start game

        # Track per-game stats for detailed logging
        game_lines_history = []
        early_failures = 0  # Games with <20 lines

        for game_number in range(self.start_game, n+1):
            tetris.reset()
            done = trained = False
            old_state = tetris.game.get_state()

            # while not done:
            #     next_states = {tuple(v): k for k, v in tetris.game.calc_all_states().items()}
            #     if not next_states:
            #         break
            #     best_state = agent.get_action(next_states.keys())
            #     lines += best_state[2]
            #     if best_state[2]==4:
            #         tetris_clears += 1
            #     best_action = next_states[best_state]
            #     # states_list, actions_list = tetris.game.calc_all_states()
            #     # if not states_list:
            #     #     break
            #     # best_idx = agent.get_action(states_list)
            #     # best_state = states_list[best_idx]
            #     # best_action = actions_list[best_idx]

            #     confidence = agent.q_values[-1] if agent.q_values else 0
            #     tetris.update_state(best_state, confidence, agent.random, agent.epsilon)

            #     reward, done = tetris.play_full(best_action)

            #     reward += self.calculate_rewards(best_state)
            #     tetris.update_rewards(reward)

            #     agent.remember(old_state, best_state, reward, done)
            #     old_state = best_state

            #     if agent.total_steps % 200 == 0:
            #         agent.train_long_memory()
            #         trained = True
            while not done:
                states_list, actions_list = tetris.game.calc_all_states()
                if not states_list:
                    break

                best_idx = agent.get_action(states_list)  # ← You already have the index here!
                best_state = states_list[best_idx]
                best_action = actions_list[best_idx]
                # DON'T add: best_idx = states_list.index(best_state)  ← Remove this!

                confidence = agent.q_values[-1] if agent.q_values else 0
                tetris.update_state(best_state, confidence, agent.random, agent.epsilon)

                reward, done = tetris.play_full(best_action)

                # Track lines after playing
                lines_removed = tetris.game.lines_removed
                lines += lines_removed
                if lines_removed == 4:
                    tetris_clears += 1

                # Extract features for reward calculation
                if self.architecture == "CNN":
                    # CNN: best_state is a board, extract features
                    features = tetris.game.get_features_from_board(best_state, lines_removed)
                else:
                    # Linear: best_state is already features [total_heights, bumpiness, lines_removed, holes, y_pos, pillar]
                    features = best_state

                reward += self.calculate_rewards(features)
                tetris.update_rewards(reward)

                agent.remember(old_state, best_state, reward, done)
                old_state = best_state

                if agent.total_steps % 200 == 0:
                    agent.train_long_memory()
                    trained = True

            if not trained:
                agent.train_long_memory()
                not_trained += 1
                if not_trained==5:
                    agent.update_target_network()
                    not_trained = 0
            else:
                not_trained = 0

            agent.decay_epsilon(tetris.games)

            tetris.games += 1
            score += tetris.game.score
            agent.calculate_lr(tetris.games)

            # Track game stats
            game_lines = tetris.game.lines
            game_lines_history.append(game_lines)
            if game_lines < 20:
                early_failures += 1

            # Log per-game metrics to wandb
            if self.use_wandb:
                wandb.log({
                    "game_number": tetris.games,
                    "game_lines": game_lines,
                    "game_score": tetris.game.score,
                    "epsilon": agent.epsilon,
                    "learning_rate": agent.LR,
                })

            # print(f'LR={agent.LR:.4f} |  Epsilon={agent.epsilon:.5f} at game={game_number}')

            if game_number % 100 == 0:
                avg_lines = lines / (game_number - self.start_game + 1)
                early_failure_rate = (early_failures / (game_number - self.start_game + 1)) * 100

                # Calculate recent performance (last 100 games)
                recent_games = game_lines_history[-100:]
                recent_avg = sum(recent_games) / len(recent_games) if recent_games else 0

                print(f'Game {game_number}/{n} | LR={agent.LR:.4f} | Epsilon={agent.epsilon:.5f} | Avg Lines={avg_lines:.2f}')

                # Log aggregate metrics every 100 games
                if self.use_wandb:
                    wandb.log({
                        "game_number": tetris.games,
                        "avg_lines_overall": avg_lines,
                        "avg_lines_recent_100": recent_avg,
                        "total_lines": lines,
                        "tetris_clears": tetris_clears,
                        "early_failure_rate": early_failure_rate,
                        "avg_score": score / (game_number - self.start_game + 1),
                        "avg_q_value": sum(agent.q_values[-100:]) / len(agent.q_values[-100:]) if agent.q_values else 0,
                    })

            if tetris.games%500==0:
                count += 1
                agent.save_model(count, tetris.games)

        # return tetris.scoreboard.hiscore, lines, tetris_clears
        return lines, tetris_clears

def run_game(SLOW_DROP=True, games=10000, resume_from=None,
             use_wandb=False, experiment_name=None, architecture="CNN",
             project_name="tetris-ai-comparison"):
    """
    Run training with optional Weights & Biases logging

    Args:
        SLOW_DROP: Whether to use slow drop mode
        games: Total number of games to train
        resume_from: Path to checkpoint to resume from
        use_wandb: Whether to use Weights & Biases logging
        experiment_name: Name for this experiment run
        architecture: "CNN" or "Linear" - which model architecture
        project_name: W&B project name
    """
    genome = {
        'game_over': 189.27613725914273,
        'survival_instinct': 8.388926084018738,
        'total_height': -0.17634932529980674,
        'lines_removed': 8.594602383216944,
        'holes': -3.743561101942274,
        'bumpiness': -6.683915232551735,
        'pillar': -11.042880500059761,
        'y_pos_reward': 207.81525814829266,
        'y_pos_punish': 117.90325502640637
    }

    n = games

    # Initialize wandb if enabled
    if use_wandb:
        config = {
            "architecture": architecture,
            "max_memory": MAX_MEMORY,
            "batch_size": BATCH_SIZE,
            "learning_rate_initial": LR,
            "learning_rate_final": 0.001,
            "epsilon_initial": 0.3,
            "epsilon_final": 0.0001,
            "gamma": 0.999,
            "total_games": games,
            "hidden_sizes": HIDDEN_SIZES,
            "epochs_per_train": EPOCHS,
            "slow_drop": SLOW_DROP,
            "reward_weights": genome,
        }

        # Add architecture-specific config
        if architecture == "CNN":
            config.update({
                "input_type": "raw_board",
                "input_shape": "(20, 10)",
                "conv_layers": 3,
                "conv_channels": [32, 64, 64],
            })
        else:
            config.update({
                "input_type": "engineered_features",
                "num_features": STATES,
            })

        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=[architecture.lower(), "ddqn", "prioritized-replay"],
            resume="allow" if resume_from else None
        )

        print(f'Weights & Biases initialized: {wandb.run.name}')

    print(f'Running simulation SLOW_DROP={SLOW_DROP}')
    if resume_from:
        print(f'Resuming from checkpoint: {resume_from}')

    start_time = time.time()
    t = Training_Simulation(genome, 1, False, n, SLOW_DROP, resume_from=resume_from, use_wandb=use_wandb, architecture=architecture)
    total_lines, total_tetris = t.run_simulation(n)
    training_time = time.time() - start_time

    # Log final summary
    if use_wandb:
        wandb.log({
            "final_total_lines": total_lines,
            "final_total_tetris_clears": total_tetris,
            "training_time_seconds": training_time,
            "training_time_hours": training_time / 3600,
        })
        wandb.finish()

    print(f'\nTraining completed in {training_time/3600:.2f} hours')
    print(f'Total lines: {total_lines} | Avg: {total_lines/games:.2f}')
    print(f'Total Tetris clears: {total_tetris}')

    return

if __name__=='__main__':
    # Example 1: Train CNN without wandb (basic training)
    # run_game(games=10000)

    # Example 2: Train CNN with wandb logging
    run_game(
        games=10000,
        use_wandb=True,
        experiment_name="cnn-tetris-v1",
        architecture="CNN"
    )

    # Example 3: Train Linear model with wandb for comparison
    # NOTE: You need to switch to Linear_QNet in agent.py first!
    # run_game(
    #     games=10000,
    #     use_wandb=True,
    #     experiment_name="linear-tetris-baseline",
    #     architecture="Linear"
    # )

    # Example 4: Resume training from checkpoint
    # run_game(
    #     games=15000,
    #     resume_from="model/trained_model_10.pth",
    #     use_wandb=True,
    #     experiment_name="cnn-tetris-v1",  # Same name to continue the run
    #     architecture="CNN"
    # )