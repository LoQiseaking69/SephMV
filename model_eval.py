import os
import gym
import numpy as np
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom RBM Layer Definition
class RBMLayer(tf.keras.layers.Layer):
    def __init__(self, num_hidden_units):
        super(RBMLayer, self).__init__()
        self.num_hidden_units = num_hidden_units

    def build(self, input_shape):
        self.rbm_weights = self.add_weight(shape=(input_shape[-1], self.num_hidden_units),
                                           initializer='random_normal',
                                           trainable=True)
        self.biases = self.add_weight(shape=(self.num_hidden_units,),
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        activation = tf.matmul(inputs, self.rbm_weights) + self.biases
        return tf.nn.sigmoid(activation)

# Custom Q-Learning Layer Definition
class QLearningLayer(tf.keras.layers.Layer):
    def __init__(self, action_space_size, state_size, learning_rate=0.01, gamma=0.95, epsilon=0.1):
        super(QLearningLayer, self).__init__()
        self.action_space_size = action_space_size
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(action_space_size, activation=None, kernel_initializer='glorot_uniform')
        ])

    def call(self, state):
        return self.q_network(state)

def load_model(model_path):
    """
    Load the saved model from a specified path.
    """
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'RBMLayer': RBMLayer, 'QLearningLayer': QLearningLayer})
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
            # Prepare the state for prediction
            state = np.array([state])
            # Predict action based on the trained model
            q_values = model(state, training=False)  # use model directly for inference
            action = np.argmax(q_values)

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