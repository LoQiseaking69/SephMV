import gym
import logging
import numpy as np
from tensorflow.keras import optimizers
from model import create_neural_network_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_in_bipedalwalker(env_name, model, num_episodes, batch_size=64):
    env = gym.make(env_name)
    for episode in range(num_episodes):
        initial_state = env.reset()
        state = np.array([initial_state])  # Ensure initial_state is a numpy array with correct shape
        done = False
        total_reward = 0
        while not done:
            action = model.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array([next_state])  # Ensure next_state is a numpy array with correct shape
            model.store_transition(state, action, reward, next_state, done)
            model.update(batch_size)
            state = next_state
            total_reward += reward
        logger.info(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')
    env.close()

env_name = 'BipedalWalker-v3'
seq_length = 128
d_model = 32
num_hidden_units = 64
action_space_size = 4
num_episodes = 1000

# Create and compile the model
model = create_neural_network_model(seq_length, d_model, num_hidden_units, action_space_size)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

try:
    train_model_in_bipedalwalker(env_name, model, num_episodes)
except Exception as e:
    logger.error(f"An error occurred during training: {e}")

# Save the trained model using the 'tf' format
model.save('Seph_model', save_format='tf')
