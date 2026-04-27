# standard imports
import numpy as np
import random
from os.path import exists

# RL imports
import gymnasium as gym
import minigrid
from minigrid.wrappers import *
from collections import namedtuple, deque

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# tensorboard imports
from torch.utils.tensorboard import SummaryWriter

# import functions
from utils import *
from train import *

### TESTING HYPERPARAMETERS
# Evaluation hyperparameter
evalEpisodes = 1000  # Number of episodes to be used for evaluation


# Set the random seed for reproducible experiments
random.seed(5)


max_steps = env.unwrapped.max_steps


def evaluate_model():
    # Evaluation loop
    print("Starting evaluation...")
    # Set the path to the saved model
    model_path = "models/trained_model.pth"

    # policy_net = DQN(inputSize, numActions, hiddenLayerSize)
    # update policy_net with model
    policy_net.load_state_dict(torch.load(model_path))

    # Make the environment
    env = create_minigrid_environment()

    # use wrapper to only extract observation
    env = ImgObsWrapper(env)

    # Initialize counters for evaluation metrics
    finishCounter = 0.0
    totalSteps = 0.0
    totalReward = 0.0

    # Run the evaluation loop
    for e in range(evalEpisodes):
        # Initialize the environment and state
        currentObs, _ = env.reset()
        currentState = preprocess(currentObs)

        # the main RL loop
        for i in range(0, max_steps):
            # Select and perform an action
            action = select_action(currentState)
            a = action.item()

            # Take action 'a', receive reward 'reward', and observe the next state 'obs'
            # 'done' indicates if the termination state was reached
            obs, reward, done, truncated, info = env.step(a)

            if done or truncated:
                # Observe the new state
                nextState = None
            else:
                nextState = preprocess(obs)

            if done or truncated:
                totalReward += reward
                totalSteps += env.unwrapped.step_count
                if done:
                    print(
                        "Finished evaluation episode %d with reward %f, %d steps, reaching the goal"
                        % (e, reward, env.unwrapped.step_count)
                    )
                    finishCounter += 1
                if truncated:
                    print(
                        "Failed evaluation episode %d with reward %f, %d steps"
                        % (e, reward, env.unwrapped.step_count)
                    )
                break

            # Move to the next state
            currentState = nextState

    # Print a summary of the evaluation results
    print(
        "Completion rate %.2f with an average reward %0.4f and average steps %0.2f"
        % (
            finishCounter / evalEpisodes,
            totalReward / evalEpisodes,
            totalSteps / evalEpisodes,
        )
    )
    print("Finished evaluation!")


if __name__ == "__main__":
    evaluate_model()
