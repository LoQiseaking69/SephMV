import os
import logging
import gym
import numpy as np
from model import create_neural_network_model

# Initialize the model with appropriate parameters
seq_length = 24  # Example value, set it according to your environment
d_model = 16     # Example value, set it according to your environment
num_hidden_units = 50  # Example value
action_space_size = 4  # Example value, set it according to the environment
model = create_neural_network_model(seq_length, d_model, num_hidden_units, action_space_size)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_in_bipedalwalker(env_name, model, num_episodes, batch_size=64):
    try:
        env = gym.make(env_name)
    except gym.error.Error as e:
        logger.error(f"Error creating environment {env_name}: {e}")
        return

    for episode in range(num_episodes):
        try:
            state = env.reset()
            state = np.array(state).reshape(1, -1)  # Reshape for model input
            done = False
            total_reward = 0

            while not done:
                action = model.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.array(next_state).reshape(1, -1)  # Reshape for model input
                model.store_transition(state, action, reward, next_state, done)
                model.update(batch_size)
                state = next_state
                total_reward += reward

            logger.info(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')
        except Exception as e:
            logger.error(f"An error occurred in episode {episode + 1}: {e}")

    env.close()

    # Save the trained model in the current directory
    save_path = 'trained_model'
    try:
        model.save(save_path, save_format='tf')
        logger.info(f"Model saved successfully at {save_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {e}")

# Example usage
env_name = 'BipedalWalker-v3'
num_episodes = 1000
train_model_in_bipedalwalker(env_name, model, num_episodes)
