import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from abc import ABC, abstractmethod
from collections import deque

from .Game import SnakeGameAI, Direction, Point
from .Model import DQN, DuelingDQN
from .ExperienceReplayBuffer import Experience, ExperienceReplayBuffer


class Agent:

    def __init__(self,
                 get_observation='relative_snake',
                 learning_rate=0.001,
                 gamma=0.9,
                 epsilon=0.9,
                 epsilon_decay=[0.999, 0.995],
                 memory_capacity=100000,
                 batch_size=1000,
                 update_frequency=20,
                 double_dqn=True,
                 dueling_dqn=False,
                 prioritized_memory=False,
                 greedy=True,
                 game = SnakeGameAI()):
        """
        Constractor for Agent class
        -----------
        Parameters:
        -----------
        get_observation (string): either 'relative_snake', 'surroundings', 'simple'
            There are three different representations of the snake's perception of
            its environment. 
                'relative_snake' defines observed state as whether the snakes body is
                ahead, right, left or behind the head, and whether there is an obstacle
                next to the snakes head.
                'surroundings' defines a 5x5 square around the snakes head and whether
                it contains walls or snake's body.
                'simple' defines whether snake is next to an obstacle and which direction
                the snake is pointing.
                All functions return whether the rat is ahead/behind or left/right of
                the snake's head.
        learning_rate (float): learning rate of the deep network between 0 and 1
        gamma (float): gamma value between 0 and 1
        epsilon (float): epsilon value between 0 and 1
        epsilon_decay (list): consists of 2 values for decay, between 0 and 1
        memory_capacity (int64): the capacity of the buffer
        update_frequency (int): the frequency of updating the target network
        double_dqn (bool): True if we want to use Double Q Learning option
        prioritized_memory (bool): True if we want to use prioritized buffer
        greedy: set to False if you want the snake to make optimal policy choices
        game: allows the game to be constructed and shared between different agents - 
              this is necessary to prevent kernal restarts
        """
        self.observation_mode = get_observation
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.prioritized_memory = prioritized_memory
        self.update_frequency = update_frequency
        self.number_episodes = 0
        self.number_timesteps = 0
        self.greedy = greedy

        self.game = game

        # TODO - get rid of prioritized_memory: we don't use it
        if prioritized_memory:
            pass
        else:
            self.replay_memory = ExperienceReplayBuffer(memory_capacity, batch_size)

        if self.observation_mode == 'relative_snake':
            self.get_observation = self.get_observation_relative_snake
            input_nodes = 12
        if self.observation_mode == 'surroundings':
            self.get_observation = self.get_observation_surroundings
            input_nodes = 29
        if self.observation_mode == 'simple':
            self.get_observation = self.get_observation_simple
            input_nodes = 11
        # define two neural network, one for policy and one for target
        if self.dueling_dqn:
            self.policy_net = DuelingDQN(input_nodes, 256, 3)
            self.target_net = DuelingDQN(input_nodes, 256, 3)
        else:
            self.policy_net = DQN(input_nodes, 256, 3)
            self.target_net = DQN(input_nodes, 256, 3)
            
        self._synchronize_q_networks()
        self.target_net.eval()

        # Define the optimizer and the loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # get observation and return np array of size 
    def get_observation_surroundings(self) -> np.array:
        """Return summary of agents observation as 
        shape (29,) np.array.
        
        Values are integers representing boolean values.
        [25 element list representing danger or safety in surrounding squares,
        rat left of snake's head,
        rat right of snake's head,
        rat above snake's head,
        rat below snake's head]
        """
        head = self.game.snake_body[0]
        block_size = self.game.block_size
        
        rat_up = self.game.rat.x < self.game.snake_head.x
        rat_down = self.game.rat.x > self.game.snake_head.x
        rat_right = self.game.rat.y > self.game.snake_head.y
        rat_left = self.game.rat.y < self.game.snake_head.y
        
        surroundings = self.game.surroundings()
        
        # depending on direction the snake is facing, rotate surroundings
        # and relative_rate arrays.
        if self.game.direction == Direction.UP:
            relative_rat = [rat_up, rat_down, rat_left, rat_right]
        if self.game.direction == Direction.LEFT:
            surroundings = np.rot90(surroundings, 1)
            relative_rat = [rat_left, rat_right, rat_down, rat_up]
        if self.game.direction == Direction.DOWN:
            surroundings = np.rot90(surroundings, 2)
            relative_rat = [rat_down, rat_up, rat_right, rat_left]
        if self.game.direction == Direction.RIGHT:
            surroundings = np.rot90(surroundings, 3)
            relative_rat = [rat_right, rat_left, rat_up, rat_down]

        return np.concatenate((surroundings.flatten(), np.array(relative_rat, dtype=int)))
    
    def get_observation_relative_snake(self) -> np.array:
        """Return summary of agents observation as 
        shape (12,) np.array.
        
        Values are integers representing boolean values.
        [4 elemnts representing if the snakes body is above, below, right or left of snake,
        4 elements representing if snake will crash into something,
        4 elements representing where the rat is relative to the snake]
        """
        rat_up = self.game.rat.x < self.game.snake_head.x
        rat_down = self.game.rat.x > self.game.snake_head.x
        rat_right = self.game.rat.y > self.game.snake_head.y
        rat_left = self.game.rat.y < self.game.snake_head.y
        
        relative_body = self.game.relative_body()
        
        relative_danger = self.game.relative_danger()
        
        # depending on direction the snake is facing, rotate relative_rat arrays.
        if self.game.direction == Direction.UP:
            relative_rat = [rat_up, rat_down, rat_left, rat_right]
        if self.game.direction == Direction.LEFT:
            relative_rat = [rat_left, rat_right, rat_down, rat_up]
        if self.game.direction == Direction.DOWN:
            relative_rat = [rat_down, rat_up, rat_right, rat_left]
        if self.game.direction == Direction.RIGHT:
            relative_rat = [rat_right, rat_left, rat_up, rat_down]

        return np.concatenate((relative_body,
                               relative_danger,
                               np.array(relative_rat, dtype=int)))
    
    # get observation and return np array of size 11
    def get_observation_simple(self) -> np.array:
        """Return summary of agents observation as 
        shape (11,) np.array.
        
        Values are integers representing boolean values.
        [danger straight, danger right, danger left,
        snake moving left, right, up, down,
        rat left of snake's head,
        rat right of snake's head,
        rat above snake's head,
        rat below snake's head]
        """
        head = self.game.snake_body[0]
        block_size = self.game.block_size

        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.game.is_collision(point_r)) or
            (dir_l and self.game.is_collision(point_l)) or
            (dir_u and self.game.is_collision(point_u)) or
            (dir_d and self.game.is_collision(point_d)),

            # Danger right
            (dir_u and self.game.is_collision(point_r)) or
            (dir_d and self.game.is_collision(point_l)) or
            (dir_l and self.game.is_collision(point_u)) or
            (dir_r and self.game.is_collision(point_d)),

            # Danger left
            (dir_d and self.game.is_collision(point_r)) or
            (dir_u and self.game.is_collision(point_l)) or
            (dir_r and self.game.is_collision(point_u)) or
            (dir_l and self.game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Rat location
            self.game.rat.x < self.game.snake_head.x,  # rat left
            self.game.rat.x > self.game.snake_head.x,  # rat right
            self.game.rat.y < self.game.snake_head.y,  # rat up
            self.game.rat.y > self.game.snake_head.y  # rat down
        ]

        return np.array(state, dtype=int)

    # method to push to the memory
    def remember(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.replay_memory.append(experience)
        # self.replay_memory.append((state, action, reward, next_state, done))  
        # popleft if memory_capacity is reached 

    # method to return random action
    def _random_action(self):
        action = [0, 0, 0]
        random_move = random.randint(0, 2)
        action[random_move] = 1
        return action

    # method to return a greedy action, takes as input state
    def _epsilon_greedy_action(self, state):
        action = [0, 0, 0]
        self.policy_net.eval()
        with torch.no_grad():
            # convert the state to a tensor:
            state = torch.tensor(state, dtype=torch.float)
            # get the prediction from the policy net:
            prediction = self.policy_net(state)  

        self.policy_net.train()
        greedy_move = torch.argmax(prediction).item()
        action[greedy_move] = 1

        return action

    # method to update the value of epsilon
    def update_policy(self):
        if self.epsilon > 0.5:
            self.epsilon *= self.epsilon_decay[0]
        else:
            self.epsilon *= self.epsilon_decay[1]

        if self.epsilon < 0.05:
            self.epsilon = 0.05

    # function to return epsilon greedy policy
    def choose_action(self, state) -> list:
        # random moves: tradeoff exploration / exploitation
        if random.uniform(0, 1) < self.epsilon and self.greedy:
            action = self._random_action()
        else:
            action = self._epsilon_greedy_action(state)
        
        # Alex: I think we really should update epsilon after each
        # episode, otherwise it's hard to compare across different 
        # algos etc cos epsilon will decay to different values over 100 episodes say,
        # and we can't tell how random the agents actions are
        # self.update_policy()
        return action

    def _synchronize_q_networks(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _soft_update_target_q_network_parameters(self) -> None:
        """Soft-update of target q-network parameters
        with the local q-network parameters."""
        for target_param, local_param in zip(self.target_net.parameters(),
                                             self.policy_net.parameters()):
            target_param.data.copy_(self.lr * local_param.data
                                    + (1 - self.lr) * target_param.data)

    # method to save the model
    def save_model(self, file_name):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({'state_dict': self.policy_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, file_name)

    # method to load the weight
    def load_model(self, file_name):
        model_path = os.path.join('./model', file_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint... ")
            # self.model.load_state_dict(torch.load(model_path))
            checkpoint = torch.load(model_path)
            self.policy_net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self._synchronize_q_networks()
            print("done !")
        else:
            print("no checkpoint found...")

    # method to return a sample from the memory
    def get_memory_sample(self):
        if self.prioritized_memory:
            pass
        else:
            mini_sample = self.replay_memory.sample()
            states, actions, rewards, next_states, dones = zip(*mini_sample)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)
        return states, actions, rewards, next_states, dones

    # Standard Q learning update
    def _q_learning_update(self, state, action, reward, next_state, done):
        # select action and evaluate it using the target network
        target = self.target_net(state)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_net(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        return target

    # Double Q learning update
    def _double_q_learning_update(self, state, action, reward, next_state, done):
        # select action using policy network and evaluate it using the target network
        target = self.policy_net(state)
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_net(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        return target

    # method to make the agent learn
    def learn(self, state, action, reward, next_state, done):
        # convert the inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # add one more dimension if the size of inputs is 1
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        if self.double_dqn:
            target = self._double_q_learning_update(state, action, reward, next_state, done)

        else:
            target = self._q_learning_update(state, action, reward, next_state, done)

        # predicted Q values with current state
        predicted = self.policy_net(state)

        self.optimizer.zero_grad()
        loss = self.criterion(target, predicted)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self._soft_update_target_q_network_parameters()

        self.number_episodes += 1
        self.number_timesteps += 1

        if self.number_timesteps % self.update_frequency == 0:
            self._synchronize_q_networks()