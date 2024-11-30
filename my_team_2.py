# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from contest.capture_agents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='QLearningOffense', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}
class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
    


NUM_TRAINING = 0
TRAINING = False

class QLearningOffense(CaptureAgent):

    def __init__(self, index):
        #parameters and weights
        super().__init__(index)
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.9
        self.numTraining = NUM_TRAINING
        self.episodesSoFar = 0
        self.weights = {
            'close_food': -3.069412528332046,
            'bias': -7.574797694683772,
            'step_ghost': -26.19208288091793,
            'food_eaten': 18.323194058137863,
            'ghost_proximity': -9.003659481660579
            #values given when training
        }

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index) #legal actions
        if not legal_actions:
            return None

        food_left = len(self.get_food(game_state).as_list())
        
        # If there are only a few food items left, prioritize finding the closest food
        if food_left <= 2:
            best_dist = 9999
            for action in legal_actions:
                successor = self.get_successor(game_state, action)
                next_pos = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, next_pos)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action


        if TRAINING: #while trainning we update weights and print them for then init the weigths with the final values printed
            for action in legal_actions:
                self.update_weights(game_state, action)
                print(f"Current weights: {self.weights}")

        if not util.flip_coin(self.epsilon):
            action = self.get_policy(game_state)
        else:
            action = random.choice(legal_actions)

        self.epsilon = max(self.epsilon * 0.99, 0.05) #we reduce exploration over time
        return action
    
   
    def get_features(self, game_state, action):
        food = self.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]

        features = util.Counter()
        features["bias"] = 1.0

        pos = game_state.get_agent_position(self.index)
        x, y = pos
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features["step_ghost"] = sum((next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        if not features["step_ghost"] and food[next_x][next_y]:
            features["food_eaten"] = 1.0

        dist = self.find_closest_food((next_x, next_y), food, walls)
        if dist is not None:
            features["close_food"] = float(dist) / (walls.width * walls.height)

        ghost_distance = min([self.get_maze_distance((next_x, next_y), g) for g in ghosts], default = 9999)
        features["ghost_proximity"] = 1 / (ghost_distance + 1)

        features.divide_all(10.0) # Normalize all the features
        return features

    def find_closest_food(self, position, food_grid, wall_grid):
        #bfs search algorithm to find which is the food that is closer
        queue = [(position[0], position[1], 0)]
        visited_positions = set()
        while queue:
            current_x, current_y, distance = queue.pop(0)
            if (current_x, current_y) in visited_positions:
                continue
            visited_positions.add((current_x, current_y))
            if food_grid[current_x][current_y]:
                return distance  
            neighbors = Actions.get_legal_neighbors((current_x, current_y), wall_grid)
            for neighbor_x, neighbor_y in neighbors:
                queue.append((neighbor_x, neighbor_y, distance + 1))
        return None  

    def get_q_value(self, game_state, action):
        features = self.get_features(game_state, action)

        return sum(features[feature] * self.weights[feature] for feature in features)

    def update(self, game_state, action, next_state, reward):
        #we update weights based on q vlaues and reward
        features = self.get_features(game_state, action)
        old_value = self.get_q_value(game_state, action)

        future_q_value = self.get_value(next_state) #max q vlaue for the next
        difference = (reward + self.discount * future_q_value) - old_value

        for feature in features:
            self.weights[feature] += self.alpha * difference * features[feature]

    def update_weights(self, game_state, action):
        #update weights for specfic action
        next_state = self.get_successor(game_state, action)
        reward = self.compute_reward(game_state, next_state)
        self.update(game_state, action, next_state, reward)

    def compute_reward(self, game_state, next_state):
        reward = 0
        pos = game_state.get_agent_position(self.index)

        if self.get_score(next_state) > self.get_score(game_state):
            diff = self.get_score(next_state) - self.get_score(game_state)
            reward = diff * 10 # increasing the score

        food = self.get_food(game_state).as_list()
        dist_to_food = min([self.get_maze_distance(pos, food) for food in food])
        if dist_to_food == 1:
            next_food = self.get_food(next_state).as_list()
            if len(food) - len(next_food) == 1:
                reward += 10 # reaward eating food

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            min_dist_ghost = min([self.get_maze_distance(pos, g.get_position()) for g in ghosts])
            if min_dist_ghost == 1:
                next_pos = next_state.get_agent_state(self.index).get_position()
                if next_pos == self.start:
                    reward -= 100 #neagtuve reward for less step ghosts

        ghost_dist = min([self.get_maze_distance(pos, g.get_position()) for g in ghosts], default= 9999)
        reward += self.weights['ghost_proximity'] * (1 / (ghost_dist + 1)) #ghost proximity neagitve reward

        return reward

    def get_successor(self, game_state, action):
        #next state after an action
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def compute_value_from_q_values(self, game_state):

        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return 0.0
        best_action = self.get_policy(game_state)
         #max q value 
        return self.get_q_value(game_state, best_action)

    def compute_action_from_q_values(self, game_state):

        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None
        
        action_q_v = {}
        best_q_v = -9999
        for action in actions:
            q_value = self.get_q_value(game_state, action)
            action_q_v[action] = q_value
            if q_value > best_q_v:
                best_q_v = q_value
        best_actions = []

        for action, value in action_q_v.items():
            if value == best_q_v:
                best_actions.append(action)

        return random.choice(best_actions)

    def get_policy(self, game_state):
        return self.compute_action_from_q_values(game_state)

    def get_value(self, game_state):
        return self.compute_value_from_q_values(game_state)


