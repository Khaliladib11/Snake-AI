import matplotlib.pyplot as plt
from IPython import display
from .Agent import Agent
import numpy as np
from .Game import SnakeGameAI

plt.ion()

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    display.clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def plot(scores, mean_scores, title=None, clear_output=True):
    if clear_output:
        display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    if not title:
        plt.title('Training...')
    else:
        plt.title(title)
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], '{:.2f}'.format(scores[-1]))
    # plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], '{:.2f}'.format(mean_scores[-1]))  
    plt.show(block=False)
    plt.pause(.1)
    
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
def ax_plot(ax, scores, mean_scores, window=100, title=None):
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    if title:
        ax.set_title(title)

    y = np.concatenate((scores[:, np.newaxis], mean_scores[:, np.newaxis]),
                       axis=1)
    
    if window:
        roll_mean = moving_average(scores, n=window)
        roll_mean = np.pad(roll_mean,
                           (len(scores) - len(roll_mean), 0),
                           'constant', constant_values=np.nan)
        y = np.concatenate((y, roll_mean[:, np.newaxis]), axis=1)
    
    ax.plot(y)
    ax.text(len(scores)-1, scores[-1], '{:.2f}'.format(scores[-1]), fontsize=8)
    ax.text(len(mean_scores)-1, mean_scores[-1], '{:.2f}'.format(mean_scores[-1]))
    
def training_loop(game_kwargs, #dict of kwargs to construct SnakeGameAI
                  model_name,  # name of model to load
                  load_model=False,
                  save_name=None, # name to save best state of model to
                  episode_save = None, # can save after n episodes
                  get_observation = 'relative_snake',
                  greedy=True,
                  epsilon_kwargs={'epsilon': 0.9, 'epsilon_decay':[0.999, 0.995]},
                  double_dqn=False, 
                  dueling_dqn=False,
                  num_episodes=1000,
                  plot_update_at_end=False,
                  random_seed=42
                 ):
    
    # to make runs with different algorithms consistent and comparible, we always
    # want the rats to appear in the same squares for each game. To achieve this
    # we have to set the random seed to be the same for the start of each episode.
    # random.randint() is also called by agent a variable amount of times in between
    # eating each rat. So we have to reset random seed after a rat is eaten.
    rng = np.random.default_rng(seed=random_seed)
    rat_seeds = rng.integers(0, 10e6, size=1000)
    
    if save_name is None:
        save_name = model_name.split('.')[0] + '_v2' + model_name.split('.')[1]
    
    game = SnakeGameAI(**game_kwargs, rat_reset_seeds=rat_seeds)
    
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    epsilons = []
    agent = Agent(double_dqn=double_dqn,
                  dueling_dqn=dueling_dqn,
                  game=game, greedy=greedy,
                  **epsilon_kwargs)
    if load_model:
        agent.load_model(model_name)
        print('loaded {}'.format(model_name))

    episode = 0
    
    while episode < num_episodes:
        state = agent.get_observation()
        action = agent.choose_action(state)
        reward, done, score = agent.game.play_step(action)
        new_state = agent.get_observation()
        # remember
        agent.remember(state, action, reward, new_state, done)

        if done:
            episode += 1
            rat_seeds = rng.integers(0, 10e6, size=1000) # new game for episode i will always
            # start the same
            agent.game.rat_reset_seeds = rat_seeds
            agent.game.reset()
            agent.update_policy()

            states, actions, rewards, new_states, dones = agent.get_memory_sample()
            agent.learn(states, actions, rewards, new_states, dones)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_episodes
            plot_mean_scores.append(mean_score)
            moving_avg = moving_average(np.array(plot_scores), 100)
            if len(moving_avg) > 0:
                if moving_avg[-1] == moving_avg.max():
                    if greedy:
                        agent.save_model(save_name)
                    
            if episode_save == episode:
                agent.save_model(save_name.split('.')[0] + '_' + str(episode_save)
                                + '_episodes.' + save_name.split('.')[1])
                    
            if plot_update_at_end and not episode == num_episodes:
                update_progress(episode/num_episodes)
                print('{}; episode: {}'.format(model_name.split('.')[0], episode))
            else:
                plot(plot_scores, plot_mean_scores)
                
    return agent, np.array(plot_scores), np.array(plot_mean_scores)