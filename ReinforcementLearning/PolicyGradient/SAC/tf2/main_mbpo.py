from environment import PyTuxActionCritic
import numpy as np
from sac_tf2 import MBPO_Agent
from utils import plot_learning_curve
import pystk
import tensorflow as tf
from tqdm import tqdm
devs = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devs[0], True)
#from gym import wrappers

def toAction(a):
    action = a.numpy()
    action = pystk.Action(*action.tolist())
    return action

if __name__ == '__main__':
    env = PyTuxActionCritic(screen_width=64, screen_height=48, verbose=True)
    agent = MBPO_Agent(input_dims=[64*48+8], env=env,
            n_actions=5, max_size=1000000, lamda=1.25)
   
    n_games = 10000 # number of episodes
    n_rollouts = 100 # number of rollouts per step
    n_updates = 20 # number of gradient updates
    k = 15 # number of rollout steps

    # uncomment this line and do a mkdir tmp && mkdir tmp/video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        # env.render(mode='human')

    tracks = ['hacienda']
    for i in range(n_games):
        # track = tracks[np.random.choice(3)]
        observation = env.reset(tracks[0])
        done = False
        score = 0

        # train model
        log_probs = agent.train_model()
        for i in range(50):
            temp = agent.train_model()
            if (temp is None):
                break
            if (abs(temp - log_probs) <= 0.1):
                break
            log_probs = temp

        # start episode
        for i in tqdm(range(1000)):
            action = agent.choose_action(observation)
            pytux_action = toAction(action)

            # take action
            _, _, done, info = env.step(pytux_action)
            # predict next state and reward
            observation_, reward = agent.predict_next(observation, action)

            loc = observation_.numpy()[64*48:64*48+3].tolist()

            # print("current distance: {:.2f} max distance: {:.2f} steps: {}".format(info, env.max_distance, env.t), end="\r")
            score += reward
            agent.record_env(observation, action, reward, observation_, done)

            # perform rollouts from sampled state(s)
            env.verbose = False
            for i in range(n_rollouts):
                obs, action, reward, obs_, done = agent.sample_env(1)
                env.set_location(obs[0,64*48:64*48+3].tolist())
                pytux_action = action[0]
                pytux_action = np.array(pytux_action).tolist()
                pytux_action = pystk.Action(*pytux_action)
                for i in range(k):
                    obs_, reward, done, info = env.step(pytux_action)
                    agent.record_model(obs, action, reward, obs_, done)
                    obs = obs_
                    action = agent.choose_action(obs)
                    pytux_action = toAction(action)

            env.set_location(loc)
            env.verbose = True
            if (done):
                break
            # perform gradient updates
            for i in range(n_updates): 
                if not load_checkpoint:
                    agent.learn()
                
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('\nepisode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

