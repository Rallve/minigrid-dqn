import numpy as np
import gymnasium as gym
from minigrid.wrappers import *
import time
import pygame
from utils import *
from train import *


def render_optimal_policy():
    """
    Renders the optimal policy for the given policy network (DQN)
    :param policy_net: the trained policy network (DQN)
    :return: None
    """

    # Evaluation loop
    print("Start render...")
    # Set the path to the saved model
    model_path = "models/new_model.pth"

    policy_net = DQN(inputSize, numActions, hiddenLayerSize)
    # update policy_net with model
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.to(device)

    # Create the environment
    env = gym.make("MiniGrid-SpikeCrossing-v0", render_mode="human")
    env = ImgObsWrapper(env)

    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption("Optimal Policy")  # Set the window title

    # Reset the environment
    obs, _ = env.reset()

    # Render the optimal policy
    done = False
    while not done:
        # Preprocess the current observation
        state = preprocess(obs)

        # Get the optimal action based on the policy network
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item()

        # Perform the action
        obs, _, done, _, _ = env.step(action)

        # Render the environment
        env.render()

        # Wait a bit
        time.sleep(0.2)

    # Close the environment
    env.close()


def main():
    # Render the optimal policy
    render_optimal_policy()


if __name__ == "__main__":
    main()
