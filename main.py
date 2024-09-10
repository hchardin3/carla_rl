import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from models.agent import Agent
from environment.carla_env import CarlaEnv
from models.cnn_model import CNN_DeepQNetwork
from utilities.engine import train_model
import os

def main():
    # 1. Load configuration from YAML files
    with open('config/carla_config.yaml', 'r') as carla_config_file:
        carla_config = yaml.safe_load(carla_config_file)

    with open('config/agent_config.yaml', 'r') as agent_config_file:
        agent_config = yaml.safe_load(agent_config_file)

    # Load the training configuration
    with open('config/training_config.yaml', 'r') as training_config_file:
        training_config = yaml.safe_load(training_config_file)

    # 2. Initialize the CARLA environment
    env = CarlaEnv(carla_config)

    # 4. Initialize the agent with the environment, state space, and action space
    agent = Agent(env=env, agent_config=agent_config)

    # 6. Set up logging directory
    log_dir = training_config.get("log_dir", "logs/")
    writer = SummaryWriter(log_dir=log_dir)

    # 7. Check if checkpoint directory exists, create it if it doesn't
    checkpoint_dir = training_config.get("checkpoint_dir", "checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 8. Training loop configuration
    num_episodes = training_config['num_episodes']
    # batch_size = training_config['batch_size']
    # max_steps_per_episode = training_config['max_steps_per_episode']
    # target_update_interval = training_config['target_update_interval']

    # Train the agent
    try:
        train_model(
            model=agent.dqn,  # DQN model
            train_dataloader=env,  # The environment will act as the source for state, actions, and rewards
            test_dataloader=env,   # If there's a separate testing environment, replace it here
            optimizer=agent.dqn.optimizer,
            loss_function=agent.dqn.loss_function,
            epochs=num_episodes,   # We use num_episodes as the number of epochs in this case
            writer=writer,
            printing=True,
            regression_problem=False
        )

        # Save the final model checkpoint
        torch.save(agent.dqn.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    finally:
        # 9. Close the environment when done
        env.close()

if __name__ == '__main__':
    main()
