# import 'gymnasium' and 'minigrid' for our environment
import gymnasium as gym
import minigrid
from minigrid.wrappers import *

# import 'random' to generate random numbers
import random

# import 'numpy' for various mathematical, vector and matrix functions
import numpy as np

from os.path import exists

# import 'Pytorch' for all our neural network needs

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Import 'namedtuple' and 'deque' for Experience Replay Memory
from collections import namedtuple, deque

# Import 'pickle' to save and load the model
import pickle

# Import 'pathlib' to check if the model exists
# from pathlib import exists

# Import 'time' to keep track of the time
import time

# if gpu is to be used, otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a named tuple to store the experiences
Transition = namedtuple("Transition", ("currentState", "action", "nextState", "reward"))


def create_minigrid_environment(grid_type="MiniGrid-Empty-8x8-v0", render_mode=None):
    """
    Create and return a MiniGrid environment.

    Parameters:
        grid_type (str): The type of grid environment to create. Defaults to 'MiniGrid-Empty-8x8-v0'.
        render_mode (str): The rendering mode. Defaults to 'none'. Set to 'human' to render the environment.

    Returns:
        gym.Env: A Gym environment object representing the MiniGrid environment.
    """
    env = gym.make(grid_type, render_mode)
    return env


def extract_object_information(obs):
    """
    Extracts the object index information from the image observation.

    The 'image' observation contains information about each tile around the agent.
    Each tile is encoded as a 3-dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE),
    where OBJECT_TO_IDX and COLOR_TO_IDX mapping can be found in 'minigrid/minigrid.py',
    and the STATE can be as follows:
        - door STATE -> 0: open, 1: closed, 2: locked

    Parameters:
        obs (numpy.ndarray): The image observation from the environment.

    Returns:
        numpy.ndarray: A 2D array containing the object index information extracted from the observation.
    """
    (rows, cols, x) = obs.shape
    tmp = np.reshape(obs, [rows * cols * x, 1], "F")[0 : rows * cols]
    return np.reshape(tmp, [rows, cols], "C")


def epsilon_greedy_action(Q, currentS_Key, numActions, epsilon):
    """
    Perform an epsilon-greedy action selection.

    Parameters:
        Q (dict): The value-function dictionary.
        currentS_Key (int): The hash key representing the current state in the value-function dictionary.
        numActions (int): The number of possible actions in the environment.
        epsilon (float): The exploration rate, indicating the probability of exploration (random action).

    Returns:
        int: The selected action.
    """

    if random.random() < epsilon:
        # Explore the environment by selecting a random action
        action = random.randint(0, numActions - 1)
    else:
        # Exploit the environment by selecting an action that maximizes the value function at the current state
        # add try catch if the action is not in the dictionary as yet. Happens when the state is actioned for the first time
        try:
            action = np.argmax(Q[currentS_Key])
        except KeyError:
            Q[currentS_Key] = np.zeros(numActions)
            action = np.argmax(Q[currentS_Key])

    return action


def normalize(observation, max_value):
    """
    Normalise the input observation so each element is a scalar value between [0,1]

    Parameters:
        observation (numpy.ndarray): The observation from the environment
        max_value (float): The maximum value to normalise the observation by

        Returns:
            numpy.ndarray: The normalised observation
    """

    return np.array(observation) / max_value


def flatten(observation):
    """
    Flatten the [7,7] observation matrix into a [1,49] tensor

    Parameters:
        observation (numpy.ndarray): The observation from the environment

    Returns:
        numpy.ndarray: The flattened observation
    """

    return torch.from_numpy(np.array(observation).flatten()).float().unsqueeze(0)


def preprocess(observation):
    """
    Combine all the preprocessing fuctions into a single function

    Parameters:
        observation (numpy.ndarray): The observation from the environment

    Returns:
        numpy.ndarray: The preprocessed observation
    """

    tensor = flatten(normalize(extract_object_information(observation), 10.0))
    return tensor.to(device)


def save_model(policy_net, filename):
    """
    Saves the model

    Parameters:
        policy_net (DQN): The policy network
        filename (str): The filename to save the model to
    """
    torch.save(policy_net.state_dict(), filename)


def load_model(filename):
    """
    Loads the model

    Parameters:
        filename (str): The filename to load the model from

    Returns:
        DQN: The policy network
    """
    if not exists(filename):
        print("Filename %s does not exist, could not load data" % filename)
        return {}

    print("Loading existing model")
    model = torch.load(filename)
    return model


class DQN(nn.Module):
    """
    Deep Q Network class

    Parameters:
        inputSize (int): The size of the flattened input state (7x7 matrix of tile IDs).
        numActions (int): The number of possible actions (left, right, move forward).
        hiddenLayerSize (tuple): The size of the hidden layers. Defaults to (512, 256).

    Attributes:
        fc1 (torch.nn.Linear): The first fully connected layer.
        fc2 (torch.nn.Linear): The second fully connected layer.
        fc3 (torch.nn.Linear): The third fully connected layer.

    Returns:
        torch.nn.Module: A PyTorch neural network module.
    """

    def __init__(self, inputSize, numActions, hiddenLayerSize=(256, 128)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], numActions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def select_action_e_greedy(
    state, stop_epsilon, start_epsilon, decay_rate, steps_done, numActions, policy_net
):
    """
    Select an action using an epsilon-greedy policy

    Parameters:
        state (torch.Tensor): The current state
        stop_epsilon (float): The epsilon value to stop decaying at
        start_epsilon (float): The epsilon value to start decaying at
        decay_rate (float): The rate at which to decay the epsilon value
        steps_done (int): The number of steps taken so far
        numActions (int): The number of possible actions
        policy_net (DQN): The policy network

    Returns:
        int: The selected action
    """

    # generate a random number
    sample = random.random()

    # calculate the epsilon threshold, based on the epsilon-start value, the epsilon-stop value,
    # the number of training steps taken and the epsilon decay rate
    # here we are using an exponential decay rate for the epsilon value
    eps_threshold = stop_epsilon + (start_epsilon - stop_epsilon) * math.exp(
        -1.0 * steps_done / decay_rate
    )

    # compare the generated random number to the epsilon threshold
    if sample > eps_threshold:
        # act greedily towards the Q-values of our policy network, given the state

        # we do not want to gather gradients as we are only generating experience, not training the network
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].unsqueeze(0)
    else:
        # select a random action with equal probability
        return torch.tensor(
            [[random.randrange(numActions)]], device=device, dtype=torch.long
        )


class ReplayMemory(object):
    """
    Replay Memory class

    Parameters:
        capacity (int): The maximum number of experiences that can be stored in memory.

    Attributes:
        memory (deque): A deque containing the experiences.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def update_target_net(policy_net, target_net):
    """
    Update the target network with the weights and biases of the policy network

    Parameters:
        policy_net (DQN): The policy network
        target_net (DQN): The target network
    """

    target_net.load_state_dict(policy_net.state_dict())


def device_specific_episodes(episodes):
    """
    Select the device to use for training and scale the number of episodes accordingly, if using GPU

    Parameters:
        episodes (int): The number of episodes to train for

    Returns:
        str: The device to use for training
    """

    # if gpu is to be used, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run for more episodes if using GPU
    if device == "cuda":
        return episodes * 10
    else:
        return episodes


def load_model(model, model_path):
    """
    Load the model

    Parameters:
        model (DQN): The model to load the weights and biases into
        model_path (str): The path to the model to load

        Returns:
            DQN: The model with the loaded weights and biases
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()
