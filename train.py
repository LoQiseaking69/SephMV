import gym
import logging
import numpy as np
from tensorflow.keras import optimizers
from model import create_neural_network_model

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
            state = np.array([state])  # Ensure state is a numpy array with correct shape
            done = False
            total_reward = 0

            while not done:
                action = model.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.array([next_state])
                model.store_transition(state, action, reward, next_state, done)
                model.update(batch_size)
                state = next_state
                total_reward += reward

            logger.info(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')
        except Exception as e:
            logger.error(f"An error occurred in episode {episode + 1}: {e}")

    env.close()

env_name = 'BipedalWalker-v3'
seq_length = 128
d_model = 32
num_hidden_units = 64
action_space_size = 4
num_episodes = 1000

try:
    model = create_neural_network_model(seq_length, d_model, num_hidden_units, action_space_size)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    train_model_in_bipedalwalker(env_name, model, num_episodes)
except Exception as e:
    logger.error(f"An error occurred during model setup or training: {e}")

# Save the trained model
try:
    model.save('Seph_model', save_format='tf')
except Exception as e:
    logger.error(f"An error occurred while saving the model: {e}")
