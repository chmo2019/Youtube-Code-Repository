import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate
from math import pi
# from tf_agents import reparameterized_sampling 

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

class ActorNetwork(keras.Model):
    def __init__(self, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

        self.mu = Dense(2, activation=None)
        self.sigma = Dense(2, activation=None)

        self.disc_mu = Dense(self.n_actions - 2, activation=None)
        self.disc_sigma = Dense(self.n_actions - 2, activation=None)


    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        disc_mu = self.disc_mu(prob)
        disc_sigma = self.disc_sigma(prob)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)
        disc_sigma = tf.clip_by_value(disc_sigma, self.noise, 1)

        return mu, sigma, disc_mu, disc_sigma

    def get_probs_and_samples(self, mu, sigma):
        samples = tf.random.normal(shape=mu.shape, mean=mu, stddev=sigma)
        probs = tf.math.log(1 / (sigma * tf.sqrt(2 * pi)) * tf.exp(-0.5 * tf.square((samples - mu) / sigma)))

        return samples, probs

    def sample_normal(self, state, reparameterize=True):
        # mu, sigma = self.call(state)
        mu, sigma, disc_mu, disc_sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        disc_probs = tfp.distributions.Normal(disc_mu, disc_sigma)

        actions, log_probs = self.get_probs_and_samples(mu, sigma)
        disc_actions, disc_log_probs = self.get_probs_and_samples(disc_mu, disc_sigma)

        # if reparameterize:
        #     actions = probabilities.sample() # + something else if you want to implement
        #     disc_actions = disc_probs.sample()
        # else:
        #     actions = probabilities.sample()
        #     disc_actions = disc_probs.sample()

        action = tf.math.tanh(actions)*self.max_action
        # log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        disc_action = tf.math.tanh(disc_actions)*self.max_action
        disc_action = tf.math.greater(disc_action, tf.cast(tf.zeros_like(disc_action), action.dtype))
        disc_action = tf.cast(disc_action, action.dtype)
        # disc_log_probs = disc_probs.log_prob(disc_actions)
        disc_log_probs -= tf.math.log(1-tf.math.pow(disc_action,2)+self.noise)
        disc_log_probs = tf.math.reduce_sum(disc_log_probs, axis=1, keepdims=True)

        action = tf.concat([action, disc_action], axis=1)
        log_probs = tf.math.add(log_probs, disc_log_probs)

        return action, log_probs
        # return tf.concat([mu, disc_mu], axis=1), log_probs

class RandomEnsemble(keras.Model):
    def __init__(self, state_shape, fc1_dims=32, n_learners=7,
            fc2_dims=32, n_actions=2, name='ensemble', chkpt_dir='tmp/mbpo'):
        ''' predicts next state and reward given current state and action'''

        super(RandomEnsemble, self).__init__()
        self.state_shape = state_shape
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_learners = n_learners
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_mbpo')
        self.noise = 1e-6

        self.fc1 = []
        self.fc2 = []
        self.state_mu = []
        self.state_sigma = []
        self.reward_mu = []
        self.reward_sigma = []
        for i in range(n_learners):
            self.fc1.append(Dense(self.fc1_dims, activation='relu'))
            self.fc2.append(Dense(self.fc2_dims, activation='relu'))
            self.state_mu.append(Dense(*self.state_shape, activation=None))
            self.state_sigma.append(Dense(*self.state_shape, activation=None))
            self.reward_mu.append(Dense(1, activation=None))
            self.reward_sigma.append(Dense(1, activation=None))

    def call(self, state, action):
        state_mu = []
        state_sigma = []
        reward_mu = []
        reward_sigma = []

        pair = tf.concat([state, action], axis=1)

        for i in range(self.n_learners):
            prob = self.fc1[i](pair)
            prob = self.fc2[i](prob)

            state_mu.append(self.state_mu[i](prob))
            state_sigma.append(self.state_sigma[i](prob))

            reward_mu.append(self.reward_mu[i](prob))
            reward_sigma.append(self.reward_sigma[i](prob))

        return state_mu, state_sigma, reward_mu, reward_sigma

    def sample_normal(self, state, action, reparameterize=True):
        # mu, sigma = self.call(state)
        idx = tf.random.uniform((), minval=0, maxval=self.n_learners-1, dtype=tf.dtypes.int32)
        state_mu, state_sigma, reward_mu, reward_sigma = self.call(state, action)
        state_mu = state_mu[idx]
        state_sigma = state_sigma[idx]
        reward_mu = reward_mu[idx]
        reward_sigma = reward_sigma[idx]

        state_probs = tfp.distributions.Normal(state_mu, state_sigma)
        reward_probs = tfp.distributions.Normal(reward_mu, reward_sigma)

        if reparameterize:
            state_preds = state_probs.sample() # + something else if you want to implement
            reward_preds = reward_probs.sample()
        else:
            state_preds = state_probs.sample()
            reward_preds = reward_probs.sample()

        state_pred = tf.math.sigmoid(state_preds)
        state_log_probs = state_probs.log_prob(state_preds)
        state_log_probs = tf.math.reduce_sum(state_log_probs, axis=1, keepdims=True)

        reward_pred = reward_preds
        reward_log_probs = state_probs.log_prob(reward_preds)
        reward_log_probs = tf.math.reduce_sum(reward_log_probs, axis=1, keepdims=True)

        log_probs = tf.math.add(state_log_probs, reward_log_probs)

        return log_probs, state_pred, reward_pred

# class CriticNetworkConv(keras.Model):
#     def __init__(self, n_actions, conv1_dims=32, conv2_dims=32, fc1_dims=32, fc2_dims=32,
#             name='critic', chkpt_dir='tmp/sac'):
#         super(CriticNetworkConv, self).__init__()
#         self.conv1_dims = conv1_dims
#         self.conv2_dims = conv2_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.model_name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

#         self.conv1 = Conv2D(self.conv1_dims, (3,3), activation='relu')
#         self.conv2 = Conv2D(self.conv2_dims, (3,3), activation='relu')

#         self.fc1 = Dense(self.fc1_dims, activation='relu')
#         self.fc2 = Dense(self.fc2_dims, activation='relu')

#         self.q = Dense(1, activation=None)

#     def call(self, state, action):
#         action_value = self.conv1(action)
#         action_value = self.conv2(action_value)
#         action_value = Flatten()(action_value)

#         state_value = self.fc1(action)
#         state_value = self.fc2(action_value)

#         action_value = tf.concat([state, action], axis=1)

#         q = self.q(action_value)

#         return q

# class ValueNetworkConv(keras.Model):
#     def __init__(self, conv1_dims=256, conv2_dims=256,
#             name='value', chkpt_dir='tmp/sac'):
#         super(ValueNetworkConv, self).__init__()
#         self.conv1_dims = conv1_dims
#         self.conv2_dims = conv2_dims
#         self.model_name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

#         self.conv1 = Conv2D(self.conv1_dims, (3,3), activation='relu')
#         self.conv2 = Conv2D(self.conv2_dims, (3,3), activation='relu')
#         self.v = Dense(1, activation=None)

#     def call(self, state):
#         state_value = self.conv1(state)
#         state_value = self.conv2(state_value)
#         state_value = Flatten()(state_value)

#         v = self.v(state_value)

#         return v

# class ActorNetworkConv(keras.Model):
#     def __init__(self, max_action, conv1_dims=256, 
#             conv2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
#         super(ActorNetworkConv, self).__init__()
#         self.conv1_dims = conv1_dims
#         self.conv2_dims = conv2_dims
#         self.n_actions = n_actions
#         self.model_name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
#         self.max_action = max_action
#         self.noise = 1e-6

#         self.conv1 = Conv2D(self.conv1_dims,(3,3), activation='relu')
#         self.conv2 = Conv2D(self.conv2_dims, (3,3), activation='relu')
#         self.mu = Dense(self.n_actions, activation=None)
#         self.sigma = Dense(self.n_actions, activation=None)

#     def call(self, state):
#         prob = self.conv1(state)
#         prob = self.conv2(prob)
#         prob = Flatten()(prob)

#         mu = self.mu(prob)
#         sigma = self.sigma(prob)
#         # might want to come back and change this, perhaps tf plays more nicely with
#         # a sigma of ~0
#         sigma = tf.clip_by_value(sigma, self.noise, 1)

#         return mu, sigma

#     def sample_normal(self, state, reparameterize=True):
#         mu, sigma = self.call(state)
#         probabilities = tfp.distributions.Normal(mu, sigma)

#         if reparameterize:
#             actions = probabilities.sample() # + something else if you want to implement
#         else:
#             actions = probabilities.sample()

#         action = tf.math.tanh(actions)*self.max_action
#         log_probs = probabilities.log_prob(actions)
#         log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
#         log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

#         return action, log_probs