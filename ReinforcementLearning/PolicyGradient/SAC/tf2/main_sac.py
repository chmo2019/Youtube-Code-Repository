from environment import PyTuxActionCritic
import numpy as np
from sac_tf2 import Agent
from utils import plot_learning_curve
import pystk
import tensorflow as tf
devs = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devs[0], True)
#from gym import wrappers

def toAction(a):
    action = a.numpy()
    action = pystk.Action(*action.tolist())
    return action

if __name__ == '__main__':
    env = PyTuxActionCritic(screen_width=64, screen_height=48, verbose=True)
    agent = Agent(input_dims=[64*48+8], env=env, batch_size=256,
            n_actions=5, max_size=1000000, lamda=2)
    # agent = AgentConv(input_dims=[96,128,3], env=env,
    #         n_actions=7, max_size=10000)
    n_games = 10000
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

    # tracks = ['hacienda', 'cocoa_temple', "zengarden"]
    for i in range(n_games):
        # track = tracks[np.random.choice(3)]
        track = "hacienda"
        observation = env.reset(track)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            pytux_action = toAction(action)
            observation_, reward, done, info = env.step(pytux_action)
            print("current distance: {:.2f} max distance: {:.2f} steps: {}".format(info, env.max_distance, env.t), end="\r")
            score += reward
            agent.remember(observation, action, reward, observation_, done)
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

