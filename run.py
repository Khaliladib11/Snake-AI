from src.Agent import Agent
from src.Game import SnakeGameAI
from src.helper import plot
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', default=10,type=int, help='The number of games.')
    parser.add_argument('--speed', default=50,type=int, help='The speed of the snake.')
    parser.add_argument('--greedy', action='store_false', help='Follow a greedy policy, set to True in training, False in testing')
    parser.add_argument('--doubleDQN', action='store_true', help='Use DoubleDQN improvement')
    parser.add_argument('--duelingDQN', action='store_true', help='Use DuelingDQN improvement.')
    parser.add_argument('--weights', default="model.pth",type=str, help='path to weights')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--UI', action='store_true', help='If you want to visualize the snake on the screen set this to True.')

    # Fetch the params from the parser
    args = parser.parse_args()
    greedy = args.greedy
    doubleDQN = args.doubleDQN
    duelingDQN = args.duelingDQN
    UI = args.UI
    nb_games = args.games
    speed = args.speed
    weights = args.weights
    plot_results = args.plot

    assert os.path.exists(os.path.join('model',weights)), f"Can't find weights at {weights}"

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    epsilons = []
    game = SnakeGameAI(UI=UI, game_speed=speed)
    agent = Agent(double_dqn=doubleDQN, dueling_dqn=duelingDQN, game=game, greedy=greedy)
    agent.load_model(weights)

    episode = 0
    NUM_EPISODES = nb_games
    while episode < NUM_EPISODES:
        state = agent.get_observation()
        action = agent.choose_action(state)
        reward, done, score = agent.game.play_step(action)
        new_state = agent.get_observation()
        # remember
        agent.remember(state, action, reward, new_state, done)
                
        if done:
            episode += 1
            # train long memory, plot result
            agent.game.reset()
            
            states, actions, rewards, new_states, dones = agent.get_memory_sample()
            agent.learn(states, actions, rewards, new_states, dones)

            if score > record:
                record = score

            

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_episodes
            plot_mean_scores.append(mean_score)
            print('Game', episode, 'Score', score, 'Record:', record, 'Mean Score: ', mean_score)
            if plot_results:
                plot(plot_scores,plot_mean_scores)


    agent.game.end_game()