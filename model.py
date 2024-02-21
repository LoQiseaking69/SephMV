# model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def sample_buffer(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        samples = np.array(random.sample(self.buffer, batch_size), dtype=object)
        return [np.stack(samples[:, i]) for i in range(samples.shape[1])]

class RBMLayer(layers.Layer):
    def __init__(self, num_hidden_units):
        super(RBMLayer, self).__init__()
        self.num_hidden_units = num_hidden_units

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("RBMLayer expects input shape of length 2")
        self.rbm_weights = self.add_weight(shape=(input_shape[-1], self.num_hidden_units),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.biases = self.add_weight(shape=(self.num_hidden_units,),
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        activation = tf.matmul(inputs, self.rbm_weights) + self.biases
        return tf.nn.sigmoid(activation)

class QLearningLayer(layers.Layer):
    def __init__(self, action_space_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        super(QLearningLayer, self).__init__()
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.buffer_index = 0
        self.replay_buffer = ReplayBuffer(100000)
        self.q_network = models.Sequential([
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.01)),  # Added another dense layer
            layers.Dense(self.action_space_size, activation='tanh', kernel_initializer='glorot_uniform')  # For continuous actions
        ])
        self.q_network.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        self.target_q_network = models.clone_model(self.q_network)

    def call(self, state):
        return self.q_network(state)

    def update(self, batch_size):
        data = self.replay_buffer.sample_buffer(batch_size)
        if data is None:
            return
        states, actions, rewards, next_states, dones = data
        target_q_values = rewards + (1 - dones) * self.gamma * np.max(self.target_q_network.predict(next_states), axis=1)
        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.q_network(states) * tf.one_hot(actions, self.action_space_size), axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        self.buffer_index += 1
        if self.buffer_index % 1000 == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition((state, action, reward, next_state, done))

    def choose_action(self, state):
      if np.random.rand() < self.epsilon:
        return np.random.randint(self.action_space_size)
      else:
        q_values = self.q_network.predict(state)
        action = np.argmax(q_values[0])
        return np.clip(action, 0, self.action_space_size - 1)


def create_neural_network_model(input_dim, num_hidden_units, action_space_size):
    input_layer = layers.Input(shape=(input_dim,))  # Adjusted for 24-dimensional input

    # A simpler architecture
    x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(input_layer)
    x = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x_rbm = RBMLayer(num_hidden_units)(x)  # Including your RBM layer
    q_learning_layer = QLearningLayer(action_space_size)(x_rbm)  # Q-Learning layer

    model = models.Model(inputs=input_layer, outputs=q_learning_layer)
    return model

# Example usage
input_dim = 24  # BipedalWalker observation space dimension
num_hidden_units = 128  # Example value
action_space_size = 4  # BipedalWalker action space dimension

model = create_neural_network_model(input_dim, num_hidden_units, action_space_size)
