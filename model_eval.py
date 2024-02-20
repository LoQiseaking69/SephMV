import os
import gym
import numpy as np
import tensorflow as tf
import logging
from model import RBMLayer, QLearningLayer, positional_encoding, transformer_encoder, create_neural_network_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load the saved model from a specified path.
    """
    try:
        custom_objects = {'RBMLayer': RBMLayer, 'QLearningLayer': QLearningLayer,
                          'positional_encoding': positional_encoding,
                          'transformer_encoder': transformer_encoder}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, env_name, num_episodes):
    """
    Evaluate the model on the environment over a number of episodes.
    """
    env = gym.make(env_name)
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = np.array([state])
            q_values = model(state, training=False)  # Use model directly for inference
            action = np.argmax(q_values[0])  # Adjusted to account for model output format

            state, reward, done, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        logger.info(f'Episode: {episode + 1}, Total Reward: {total_reward}')

    env.close()
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    logger.info(f'Average Reward over {num_episodes} episodes: {avg_reward}')
    logger.info(f'Standard Deviation of Reward: {std_reward}')
    return avg_reward, std_reward

def main():
    model_directory = os.getenv('MODEL_PATH')
    if not model_directory:
        raise ValueError("MODEL_PATH environment variable not set")

    env_name = 'BipedalWalker-v3'
    num_episodes = 100  # Number of episodes to evaluate

    model_path = os.path.join(model_directory, 'Seph_model')  # Adjusted to the SavedModel directory
    model = load_model(model_path)
    evaluate_model(model, env_name, num_episodes)

if __name__ == "__main__":
    main()
