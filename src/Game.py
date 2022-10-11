import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys

pygame.init()
font = pygame.font.Font(None, 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class SnakeGameAI:

    def __init__(self, width=640, height=480, block_size=20,
                 UI=False,
                 game_speed=100, window_title="RL Snake",
                 rat_reset_seeds=np.random.randint(0,100000,size=1000)
                ):
        """
        AI Snake Game Environment
        :param width (int): width of the game window
        :param height (int): height of the game window
        :param block_size (int): block size of each block on the window
        :param UI (bool): set to true to display pyGame UI. Can set to
            false if training in cloud environment with no video UI
        :param game_speed (int): the speed of the game
        :param window_title (str): the title of the window
        :param seed (int): random seed
        """
        self._width = width
        self._height = height
        self._block_size = block_size
        self._game_speed = game_speed
        self._rat_reset_seeds = iter(rat_reset_seeds)
        self.UI = UI
        # init display
        if self.UI:
            self.display = pygame.display.init()
            self.display = pygame.display.set_mode((self._width, self._height))
            pygame.display.set_caption(window_title)
            self.clock = pygame.time.Clock()

        self.direction = None
        self.snake_head = None
        self.snake_body = None
        self.rat = None
        self.score = 0
        self.frame_iteration = 0
        self.reset()

    # Getter and setter
    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        if width > 0:
            self._width = width

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        if height > 0:
            self._height = height

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, block_size):
        if block_size > 0:
            self._block_size = block_size

    @property
    def game_speed(self):
        return self._game_speed

    @game_speed.setter
    def game_speed(self, game_speed):
        if game_speed > 0:
            self._game_speed = game_speed
            
    @property
    def rat_reset_seeds(self):
        return self._rat_reset_seeds

    @rat_reset_seeds.setter
    def rat_reset_seeds(self, seed_array):
         self._rat_reset_seeds = iter(seed_array)

    # method to reset the environment
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.snake_head = Point(self._width / 2, self._height / 2)
        self.snake_body = [self.snake_head,
                           Point(self.snake_head.x
                                 - self._block_size, self.snake_head.y),
                           Point(self.snake_head.x
                                 - (2 * self._block_size), self.snake_head.y)]

        self.score = 0
        self.rat = None
        self._place_rat()
        self.frame_iteration = 0

    # method to place a rat randomly on the screen
    def _place_rat(self):
        random.seed(next(self._rat_reset_seeds))
        x = random.randint(0, (self._width - self._block_size)
                           // self._block_size) * self._block_size
        y = random.randint(0, (self._height - self._block_size)
                           // self._block_size) * self._block_size
        self.rat = Point(x, y)
        if self.rat in self.snake_body:
            self._place_rat()

    # quit the game
    def end_game(self):
        pygame.quit()

    # method to place a step based on action
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        if self.UI:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_game()

        # 2. move
        self._move(action)  # update the head
        self.snake_body.insert(0, self.snake_head)

        # 3. check if game over
        reward = 0
        game_over = False
        if (self.is_collision()
            # what does this do??
            or self.frame_iteration > 100 * len(self.snake_body)):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.snake_head == self.rat:
            self.score += 1
            reward = 10
            self._place_rat()
        else:
            self.snake_body.pop()

        # 5. update ui and clock
        if self.UI:
            self._update_ui()
            self.clock.tick(self._game_speed)
        # 6. return game over and score
        return reward, game_over, self.score

    # method to check if there is a collision or not
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake_head
        # hits boundary
        if (pt.x > self._width - self._block_size
            or pt.x < 0 or pt.y > self._height - self._block_size
            or pt.y < 0):
            return True
        # hits itself
        if pt in self.snake_body[1:]:
            return True

        return False
    
    def relative_danger(self):
        """returns flattend array of shape (4,)
        Indices represent [ahead, right, left, behind]
        Values are one if collision is in that direction
        """
        # First create ndarray of shape (2,2) to represent absolute position
        # of collision relative to head
        # indices represent: [[up, right],
        #                     [left, down]]
        point_l = Point(self.snake_head.x - self.block_size, self.snake_head.y)
        point_r = Point(self.snake_head.x + self.block_size, self.snake_head.y)
        point_u = Point(self.snake_head.x, self.snake_head.y - self.block_size)
        point_d = Point(self.snake_head.x, self.snake_head.y + self.block_size)
        
        A = np.zeros((2,2))
        
        if self.is_collision(point_u):
            A[0,0] = 1
        if self.is_collision(point_d):
            A[1,1] = 1
        if self.is_collision(point_l):
            A[1,0] = 1
        if self.is_collision(point_r):
            A[0,1] = 1
        
        # TODO: code below is repeated in relative_body() should make into its
        # own method to prevent repitition.
        
        # R is position of snakes body RELATIVE to direction snake is moving
        # To transform A to R, rotate A depeding on direction snake is moving
        if self.direction == Direction.UP:
            R = A
        if self.direction == Direction.LEFT:
            R = np.rot90(A, 1)
        if self.direction == Direction.DOWN:
            R = np.rot90(A, 2)
        if self.direction == Direction.RIGHT:
            R = np.rot90(A, 3)
            
        return R.flatten()
            
    
    def relative_body(self):
        """Returns ndarray of shape (4,)
        Indices represent [ahead, right, left, behind]
        Values are one if a segment of the snakes body is in that direction relative
        to the snake, zero if not.
        """
        # First create ndarray of shape (2,2) to represent absolute position
        # of snake body relative to head
        # indices represent: [[up, right],
        #                     [left, down]]
        A = np.zeros((2,2))
        for segment in self.snake_body[1:]:
            if segment == self.snake_head.x:
                if segment < self.snake_head.y:
                    A[0,0] = 1
                if segment > self.snake_head.y:
                    A[1,1] = 1
            if segment == self.snake_head.y:
                if segment < self.snake_head.x:
                    A[1,0] = 1
                if segment > self.snake_head.x:
                    A[0,1] = 1
        
        # R is position of snakes body RELATIVE to direction snake is moving
        # To transform A to R, rotate A depeding on direction snake is moving
        if self.direction == Direction.UP:
            R = A
        if self.direction == Direction.LEFT:
            R = np.rot90(A, 1)
        if self.direction == Direction.DOWN:
            R = np.rot90(A, 2)
        if self.direction == Direction.RIGHT:
            R = np.rot90(A, 3)
            
        return R.flatten()

    # method to update the user interface
    def _update_ui(self):
        
        # snake body parts are dark blue squares (BLUE1) with
        # light blue borders (BLUE2)
        # calculate adjustment for BLUE2
        border = self._block_size // 5
        # size of each side of inner square
        border2 = self._block_size - 2 * border
        
        self.display.fill(BLACK)

        for pt in self.snake_body:
            pygame.draw.rect(self.display,
                             BLUE1,
                             pygame.Rect(pt.x, pt.y,
                                         self._block_size,
                                         self._block_size))
            pygame.draw.rect(self.display,
                             BLUE2,
                             pygame.Rect(pt.x + border,
                                         pt.y + border,
                                         border2, border2))

        pygame.draw.rect(self.display,
                         RED,
                         pygame.Rect(self.rat.x, self.rat.y,
                                     self._block_size,
                                     self._block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # method to move based on action
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.snake_head.x
        y = self.snake_head.y
        if self.direction == Direction.RIGHT:
            x += self._block_size
        elif self.direction == Direction.LEFT:
            x -= self._block_size
        elif self.direction == Direction.DOWN:
            y += self._block_size
        elif self.direction == Direction.UP:
            y -= self._block_size

        self.snake_head = Point(x, y)

    def _create_env_array(self):
        """Change environment to an array where there are padded boundaries
        and snake body parts. Anything that kills snakes with collision is
        represented by a 1, anything that doesn't is represented as 0
        
        We pad with 3 rows/columns of walls. This is because a terminal state
        give snake location as part of the wall. To avoid errors, x or y +- 2
        has to be an index in the array. 
        """
        ar_height = int(self.height / self.block_size)
        ar_width = int(self.width / self.block_size)
        
        self.env_array = np.ones((ar_width + 6, ar_height + 6))
        self.env_array[3:-3, 3:-3] = 0
        
        for point in self.snake_body:
            x, y = self._coord_to_ar_idx(point)
            self.env_array[x,y] = 1
        
    def _coord_to_ar_idx(self, point):
        """Convert x,y coord to array index.
        
        Devide by block size and add 2.
        
        E.g. if x = 0, snake is still in the game. So with grid padded by 2, 
        x index has to be = 2. 
        """
        return (int(point.x / self.block_size) + 3,
                int(point.y / self.block_size) + 3)
    
    def surroundings(self):
        """Return 5x5 array of immediate surroundings"""
        
        self._create_env_array()
        x, y = self._coord_to_ar_idx(self.snake_head)
        surroundings = self.env_array[x - 2: x +3,
                                      y - 2: y + 3]
        # get_observations method creates list and converts to array.
        # may as well supply it with a list from .surroundings()
        
        return surroundings