import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork, RandomEnsemble

tf.keras.backend.set_floatx('float32')

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], lamda=1,
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', 
                                    max_action=1)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(name='value')
        self.target_value = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.lamda = lamda

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)

    @tf.function
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs *= self.lamda
            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))


        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            log_probs *= self.lamda
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))
        

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()


class MBPO_Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], lamda=1,
            env=None, gamma=0.99, n_actions=2, max_size=10000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau

        self.env_memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.model_memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', 
                                    max_action=1)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1', chkpt_dir='tmp/mbpo')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2', chkpt_dir='tmp/mbpo')
        self.value = ValueNetwork(name='value', chkpt_dir='tmp/mbpo')
        self.target_value = ValueNetwork(name='target_value', chkpt_dir='tmp/mbpo')
        self.ensemble = RandomEnsemble(state_shape=input_dims, name='ensemble', chkpt_dir='tmp/mbpo')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))
        self.ensemble.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.lamda = lamda

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions[0]

    def predict_next(self, observation, action):
        state = tf.convert_to_tensor([observation])
        action = tf.convert_to_tensor([action])

        _, next_state_pred, reward_pred = self.ensemble.sample_normal(state, action)

        return next_state_pred[0], reward_pred[0]

    def record_model(self, state, action, reward, new_state, done):
        self.model_memory.store_transition(state, action, reward, new_state, done)

    def record_env(self, state, action, reward, new_state, done):
        self.env_memory.store_transition(state, action, reward, new_state, done)

    def sample_model(self, batch_size):
        return self.model_memory.sample_buffer(batch_size)

    def sample_env(self, batch_size):
        return self.env_memory.sample_buffer(batch_size)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)
        self.ensemble.save_weights(self.ensemble.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.value.load_weights(self.value.checkpoint_file)
        self.target_value.load_weights(self.target_value.checkpoint_file)
        self.ensemble_value.load_weights(self.ensemble.checkpoint_file)

    @tf.function
    def train_model(self):
        if self.env_memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.env_memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # update with negative log likelihood
        with tf.GradientTape() as tape:
            log_probs, next_state_pred, reward_pred = \
                self.ensemble.sample_normal(states, actions)

            ensemble_loss = -log_probs

        ensemble_network_gradient = tape.gradient(ensemble_loss, 
            self.ensemble.trainable_variables)

        self.ensemble.optimizer.apply_gradients(zip(
            ensemble_network_gradient, self.ensemble.trainable_variables))

        return log_probs

    @tf.function
    def learn(self):
        if self.model_memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.model_memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # _, states_, rewards = self.ensemble.sample_normal(states, actions)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs *= self.lamda
            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))


        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            log_probs *= self.lamda
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))
        

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()

# class AgentConv:
#     def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
#             env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
#             layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
#         self.gamma = gamma
#         self.tau = tau
#         self.memory = ReplayBuffer(max_size, input_dims, n_actions)
#         self.batch_size = batch_size
#         self.n_actions = n_actions

#         self.actor = ActorNetworkConv(n_actions=n_actions, name='actor', 
#                                     max_action=1)
#         self.critic_1 = CriticNetworkConv(n_actions=n_actions, name='critic_1')
#         self.critic_2 = CriticNetworkConv(n_actions=n_actions, name='critic_2')
#         self.value = ValueNetworkConv(name='value')
#         self.target_value = ValueNetworkConv(name='target_value')

#         self.actor.compile(optimizer=Adam(learning_rate=alpha))
#         self.critic_1.compile(optimizer=Adam(learning_rate=beta))
#         self.critic_2.compile(optimizer=Adam(learning_rate=beta))
#         self.value.compile(optimizer=Adam(learning_rate=beta))
#         self.target_value.compile(optimizer=Adam(learning_rate=beta))

#         self.scale = reward_scale
#         self.update_network_parameters(tau=1)

#     def choose_action(self, observation):
#         state = tf.convert_to_tensor([observation])
#         actions, _ = self.actor.sample_normal(state, reparameterize=False)

#         return actions[0]

#     def remember(self, state, action, reward, new_state, done):
#         self.memory.store_transition(state, action, reward, new_state, done)

#     def update_network_parameters(self, tau=None):
#         if tau is None:
#             tau = self.tau

#         weights = []
#         targets = self.target_value.weights
#         for i, weight in enumerate(self.value.weights):
#             weights.append(weight * tau + targets[i]*(1-tau))

#         self.target_value.set_weights(weights)

#     def save_models(self):
#         print('... saving models ...')
#         self.actor.save_weights(self.actor.checkpoint_file)
#         self.critic_1.save_weights(self.critic_1.checkpoint_file)
#         self.critic_2.save_weights(self.critic_2.checkpoint_file)
#         self.value.save_weights(self.value.checkpoint_file)
#         self.target_value.save_weights(self.target_value.checkpoint_file)

#     def load_models(self):
#         print('... loading models ...')
#         self.actor.load_weights(self.actor.checkpoint_file)
#         self.critic_1.load_weights(self.critic_1.checkpoint_file)
#         self.critic_2.load_weights(self.critic_2.checkpoint_file)
#         self.value.load_weights(self.value.checkpoint_file)
#         self.target_value.load_weights(self.target_value.checkpoint_file)

#     def learn(self):
#         if self.memory.mem_cntr < self.batch_size:
#             return

#         state, action, reward, new_state, done = \
#                 self.memory.sample_buffer(self.batch_size)

#         states = tf.convert_to_tensor(state, dtype=tf.float32)
#         states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
#         rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
#         actions = tf.convert_to_tensor(action, dtype=tf.float32)

#         with tf.GradientTape() as tape:
#             value = tf.squeeze(self.value(states), 1)
#             value_ = tf.squeeze(self.target_value(states_), 1)

#             current_policy_actions, log_probs = self.actor.sample_normal(states,
#                                                         reparameterize=False)
#             log_probs = tf.squeeze(log_probs,1)
#             q1_new_policy = self.critic_1(states, current_policy_actions)
#             q2_new_policy = self.critic_2(states, current_policy_actions)
#             critic_value = tf.squeeze(
#                                 tf.math.minimum(q1_new_policy, q2_new_policy), 1)

#             value_target = critic_value - log_probs
#             value_loss = 0.5 * keras.losses.MSE(value, value_target)

#         value_network_gradient = tape.gradient(value_loss, 
#                                                 self.value.trainable_variables)
#         self.value.optimizer.apply_gradients(zip(
#                        value_network_gradient, self.value.trainable_variables))


#         with tf.GradientTape() as tape:
#             # in the original paper, they reparameterize here. We don't implement
#             # this so it's just the usual action.
#             new_policy_actions, log_probs = self.actor.sample_normal(states,
#                                                 reparameterize=True)
#             log_probs = tf.squeeze(log_probs, 1)
#             q1_new_policy = self.critic_1(states, new_policy_actions)
#             q2_new_policy = self.critic_2(states, new_policy_actions)
#             critic_value = tf.squeeze(tf.math.minimum(
#                                         q1_new_policy, q2_new_policy), 1)
        
#             actor_loss = log_probs - critic_value
#             actor_loss = tf.math.reduce_mean(actor_loss)

#         actor_network_gradient = tape.gradient(actor_loss, 
#                                             self.actor.trainable_variables)
#         self.actor.optimizer.apply_gradients(zip(
#                         actor_network_gradient, self.actor.trainable_variables))
        

#         with tf.GradientTape(persistent=True) as tape:
#             # I didn't know that these context managers shared values?
#             q_hat = self.scale*reward + self.gamma*value_*(1-done)
#             q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
#             q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
#             critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
#             critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
#         critic_1_network_gradient = tape.gradient(critic_1_loss,
#                                         self.critic_1.trainable_variables)
#         critic_2_network_gradient = tape.gradient(critic_2_loss,
#             self.critic_2.trainable_variables)

#         self.critic_1.optimizer.apply_gradients(zip(
#             critic_1_network_gradient, self.critic_1.trainable_variables))
#         self.critic_2.optimizer.apply_gradients(zip(
#             critic_2_network_gradient, self.critic_2.trainable_variables))

#         self.update_network_parameters()