import torch as T
import numpy as np
from collections import deque
from models.cnn_model import CNN_DeepQNetwork
import random
from environment.carla_env import CarlaEnv

class Agent:
    def __init__(self, env: CarlaEnv, agent_config: dict):
        """
        Initializes the Agent with an environment and a configuration dictionary.

        Parameters:
        - env (CarlaEnv): The CARLA environment for the agent to interact with.
        - agent_config (dict): Dictionary containing the agent's hyperparameters.
            - 'state_space': State space dimension.
            - 'action_space': Action space dimension.
            - 'max_memory': Maximum size of the replay buffer.
            - 'learning_rate': Learning rate for the optimizer.
            - 'discount_factor': Discount factor (gamma) for the Q-learning update.
            - 'epsilon': Initial value of epsilon for epsilon-greedy exploration.
            - 'epsilon_decay': Rate at which epsilon decays after each episode.
            - 'epsilon_min': Minimum value of epsilon.
            - 'target_update_interval': How often to update the target network.
        """
        self.env = env
        self.state_space = agent_config['state_space']
        self.action_space = agent_config['action_space']
        self.discount_factor = agent_config['discount_factor']
        self.learning_rate = agent_config['learning_rate']

        self.memory = deque(maxlen=int(agent_config['max_memory']))
        self.epsilon = agent_config['epsilon']
        self.epsilon_decay = agent_config['epsilon_decay']
        self.epsilon_min = agent_config['epsilon_min']
        self.target_update_interval = agent_config['target_update_interval']  # Added

        self.step = 0

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Initialize DQNs
        self.dqn = CNN_DeepQNetwork(self.state_space, self.action_space).to(self.device)
        self.target_dqn = CNN_DeepQNetwork(self.state_space, self.action_space).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        state = T.tensor([state], dtype=T.float).to(self.device)
        rand = np.random.random()

        if rand < self.epsilon:
            # Exploration: Random action within the valid ranges for the continuous action space
            action = np.array([
                np.random.uniform(-self.env.back_acc, self.env.max_acc),  # Throttle (backward or forward)
                np.random.uniform(0, self.env.max_decc),                  # Brake
                np.random.uniform(-self.env.max_wheel_angle, self.env.max_wheel_angle)  # Steering
            ])
        else:
            # Exploitation: Select action based on Q-values from the network
            actions = self.dqn(state)
            
            # Assuming the network outputs a set of continuous values, directly use them
            action = actions.cpu().detach().numpy().squeeze()

            # Make sure the action values fall within the allowed action ranges
            action[0] = np.clip(action[0], -self.env.back_acc, self.env.max_acc)  # Throttle
            action[1] = np.clip(action[1], 0, self.env.max_decc)                  # Brake
            action[2] = np.clip(action[2], -self.env.max_wheel_angle, self.env.max_wheel_angle)  # Steering

        self.step += 1
        return action

    def learn(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        self.dqn.optimizer.zero_grad()

        # Sample batch from memory
        batch = random.sample(self.memory, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = T.tensor(state_batch, dtype=T.float).to(self.device)
        action_batch = T.tensor(action_batch).to(self.device)
        reward_batch = T.tensor(reward_batch, dtype=T.float).to(self.device)
        next_state_batch = T.tensor(next_state_batch, dtype=T.float).to(self.device)
        done_batch = T.tensor(done_batch, dtype=T.float).to(self.device)

        # Q-values for the current state-action pairs
        q_values = self.dqn(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

        # Q-values for the next state using the target network
        next_q_values = self.target_dqn(next_state_batch).max(1)[0].detach()

        # Calculate target Q-values
        target_q_values = reward_batch + self.discount_factor * next_q_values * (1 - done_batch)

        # Loss calculation
        loss = self.dqn.loss_function(q_values, target_q_values)
        loss.backward()

        # Gradient clipping
        T.nn.utils.clip_grad_norm_(self.dqn.parameters(), max_norm=1.0)

        # Optimizer step
        self.dqn.optimizer.step()

        # Apply the learning rate scheduler step
        self.dqn.scheduler.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update the target network at specified intervals
        if self.step % self.target_update_interval == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
