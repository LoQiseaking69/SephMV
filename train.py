import os
import logging
import gym
import numpy as np
from model import create_neural_network_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_in_bipedalwalker(env_name, q_learning_layer, num_episodes, epsilon=0.1):
    env = gym.make(env_name)

    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state).reshape(1, -1)
        done = False
        total_reward = 0

        while not done:
            action = q_learning_layer.choose_action(state)  # Use Q-learning layer to choose action

            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)

            # Assuming store_transition and update are methods of the QLearningLayer
            q_learning_layer.store_transition(state, action, reward, next_state, done)
            q_learning_layer.update(batch_size=32)  # Update the Q-learning layer with a batch size of 32

            state = next_state
            total_reward += reward

        logger.info(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')

    env.close()

    # Save the trained model
    save_path = 'trained_model.h5'
    q_learning_layer.save_weights(save_path)
    logger.info(f"Model saved successfully at {save_path}")

# Example usage
env_name = 'BipedalWalker-v3'
num_episodes = 1000
epsilon = 0.1  # Define epsilon value for exploration

q_learning_layer = QLearningLayer(action_space_size)
train_model_in_bipedalwalker(env_name, q_learning_layer, num_episodes, epsilon=epsilon)
