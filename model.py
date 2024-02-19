import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import gym
from collections import deque

# ReplayBuffer implementation
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def sample_buffer(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
        # Ensuring all arrays are of consistent shape
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones

# RBM Layer definition
class RBMLayer(layers.Layer):
    def __init__(self, num_hidden_units):
        super(RBMLayer, self).__init__()
        self.num_hidden_units = num_hidden_units

    def build(self, input_shape):
        self.rbm_weights = self.add_weight(shape=(input_shape[-1], self.num_hidden_units),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.biases = self.add_weight(shape=(self.num_hidden_units,),
                                      initializer='zeros',
                                      trainable=True)

    def call(self, inputs):
        activation = tf.matmul(inputs, self.rbm_weights) + self.biases
        return tf.nn.sigmoid(activation)

# QLearningLayer definition
class QLearningLayer(layers.Layer):
    def __init__(self, action_space_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        super(QLearningLayer, self).__init__()
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = ReplayBuffer(100000)

    def build(self, input_shape):
        self.q_network = models.Sequential([
            layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(self.action_space_size, kernel_initializer='glorot_uniform')
        ])
        self.q_network.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

    def call(self, state):
        return self.q_network(state)

    def update(self, batch_size):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_buffer(batch_size)
        q_values = self.q_network.predict(states)
        next_q_values = self.q_network.predict(next_states)

        target_q_values = np.copy(q_values)
        batch_indices = np.arange(batch_size, dtype=np.int32)

        target_q_values[batch_indices, actions] = rewards + (1 - dones) * self.gamma * np.max(next_q_values, axis=1)

        self.q_network.fit(states, target_q_values, epochs=1, verbose=0)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            q_values = self.q_network.predict(state)
            return np.argmax(q_values)

# Positional Encoding for Transformer
def positional_encoding(seq_length, d_model):
    position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
    sine_terms = tf.sin(position * div_term)
    cosine_terms = tf.cos(position * div_term)
    pos_encoding = tf.concat([sine_terms, cosine_terms], axis=-1)
    return pos_encoding

# Transformer Encoder Layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation="relu", kernel_initializer='he_uniform')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1], kernel_initializer='glorot_uniform')(x)
    return x + res

# Model building function with integrated RBM, Q-Learning, LSTM, Conv1D, and Transformer layers
def create_neural_network_model(seq_length, d_model, num_hidden_units, action_space_size):
    input_layer = layers.Input(shape=(seq_length, d_model))
    x_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_initializer='glorot_uniform'))(input_layer)
    x_conv = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x_lstm)
    x_rbm = RBMLayer(num_hidden_units)(x_conv)  # Pass the output of Conv1D directly to RBMLayer
    q_learning_layer = QLearningLayer(action_space_size)(x_rbm)
    model = models.Model(inputs=input_layer, outputs=q_learning_layer)
    return model

# Training the model in the BipedalWalker environment
def train_model_in_bipedalwalker(env_name, model, num_episodes, batch_size=64):
    env = gym.make(env_name)
    for episode in range(num_episodes):
        initial_state = env.reset()
        state = np.expand_dims(initial_state, axis=0)
        done = False
        total_reward = 0
        while not done:
            action = model.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            model.store_transition(state, action, reward, next_state, done)
            model.update(batch_size)
            state = next_state
            total_reward += reward
        print(f'Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}')
    env.close()

# Parameters for the model and training
env_name = 'BipedalWalker-v3'
seq_length = 128
d_model = 32
num_hidden_units = 64
action_space_size = 4
num_episodes = 1000

# Create and compile the model
model = create_neural_network_model(seq_length, d_model, num_hidden_units, action_space_size)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model
train_model_in_bipedalwalker(env_name, model, num_episodes)

# Save the trained model
model.save('Seph_model.h5')
