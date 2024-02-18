import os
import gym
import numpy as np
import tensorflow as tf
import logging

def load_model(model_path):
    """
    Load the saved model from a specified path.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, env_name, num_episodes):
    """
    Evaluate the model on the environment over a number of episodes.
    """
    try:
        env = gym.make(env_name)
        total_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = model.choose_action(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

            total_rewards.append(total_reward)
            logging.info(f'Episode: {episode + 1}, Total Reward: {total_reward}')

        env.close()
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        logging.info(f'Average Reward over {num_episodes} episodes: {avg_reward}')
        logging.info(f'Standard Deviation of Reward: {std_reward}')
        return avg_reward, std_reward

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def main():
    logging.basicConfig(level=logging.INFO)
    model_directory = os.getenv('GITHUB_WORKSPACE')
    if not model_directory:
        raise ValueError("GITHUB_WORKSPACE environment variable not set")

    model_files = [f for f in os.listdir(model_directory) if f.endswith('.h5')]
    if not model_files:
        raise FileNotFoundError("No .h5 model file found in the specified directory")

    model_path = os.path.join(model_directory, model_files[0])
    env_name = 'BipedalWalker-v3'
    num_episodes = 100  # Number of episodes to evaluate

    try:
        model = load_model(model_path)
        average_reward, std_reward = evaluate_model(model, env_name, num_episodes)
        # Further processing of average_reward and std_reward as needed
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
