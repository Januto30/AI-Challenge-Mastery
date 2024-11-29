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

NUM_TRAINING = 0
TRAINING = False

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ApproxQLearningOffense', second='DefensiveReflexAgent', num_training=0):
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





class ApproxQLearningOffense(CaptureAgent):
    
    def register_initial_state(self, game_state):
        """
        Initializes the agent's learning parameters and other necessary attributes.
        """
        self.epsilon = 0.1
        self.alpha = 0.2
        self.discount = 0.9
        self.numTraining = NUM_TRAINING
        self.episodesSoFar = 0
        
        # Set up initial weights
        self.weights = {
            'closest-food': -3.099192562140742,
            'bias': -9.280875042529367,
            '#-of-ghosts-1-step-away': -16.6612110039328,
            'eats-food': 11.127808437648863,
            'ghost-distance': -5.0,  # New feature: Ghost distance
            'score-difference': 2.0  # New feature: Score difference
        }

        self.start = game_state.get_agent_position(self.index)
        self.featuresExtractor = FeaturesExtractor(self)
        super().register_initial_state(game_state)
        
    def choose_action(self, game_state):
        """
        Choose the best action based on the current Q-values.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None

        food_left = len(self.get_food(game_state).as_list())
        
        # If there are only a few food items left, prioritize finding the closest food
        if food_left <= 2:
            best_dist = float('inf')
            for action in legal_actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        # During training, update weights
        action = None
        if TRAINING:
            for action in legal_actions:
                self.update_weights(game_state, action)
        
        # Exploration vs exploitation with epsilon decay
        if not util.flip_coin(self.epsilon):
            action = self.get_policy(game_state)
        else:
            action = random.choice(legal_actions)
        
        # Decay epsilon over time to shift from exploration to exploitation
        self.epsilon = max(self.epsilon * 0.99, 0.05)  # Epsilon decay

        return action

    def get_weights(self):
        """
        Returns the current weight vector.
        """
        return self.weights

    def get_q_value(self, game_state, action):
        """
        Calculates Q(state, action) = w * featureVector
        """
        features = self.featuresExtractor.get_features(game_state, action)
        return sum(features[feature] * self.weights[feature] for feature in features)

    def update(self, game_state, action, next_state, reward):
        """
        Updates weights based on the given transition.
        """
        features = self.featuresExtractor.get_features(game_state, action)
        old_value = self.get_q_value(game_state, action)
        future_q_value = self.get_value(next_state)
        difference = (reward + self.discount * future_q_value) - old_value
        
        # Update weights for each feature
        for feature in features:
            new_weight = self.alpha * difference * features[feature]
            self.weights[feature] += new_weight

    def update_weights(self, game_state, action):
        """
        Updates weights after each action.
        """
        next_state = self.get_successor(game_state, action)
        reward = self.get_reward(game_state, next_state)
        self.update(game_state, action, next_state, reward)

    def get_reward(self, game_state, next_state):
        """
        Computes the reward for the agent given the current and next state.
        """
        reward = 0
        agent_position = game_state.get_agent_position(self.index)

        # Check if the agent has updated the score (i.e., scored a point)
        if self.get_score(next_state) > self.get_score(game_state):
            diff = self.get_score(next_state) - self.get_score(game_state)
            reward = diff * 10

        # Check if the agent eats food in the next state
        my_foods = self.get_food(game_state).as_list()
        dist_to_food = min([self.get_maze_distance(agent_position, food) for food in my_foods])
        if dist_to_food == 1:
            next_foods = self.get_food(next_state).as_list()
            if len(my_foods) - len(next_foods) == 1:
                reward += 10  # Reward for eating food

        # Check if the agent is eaten (i.e., ghost is too close)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if len(ghosts) > 0:
            min_dist_ghost = min([self.get_maze_distance(agent_position, g.get_position()) for g in ghosts])
            if min_dist_ghost == 1:
                next_pos = next_state.get_agent_state(self.index).get_position()
                if next_pos == self.start:
                    reward -= 100  # Negative reward for returning to start when in danger

        # Additional reward for being closer to the ghosts or farther away
        ghost_dist = min([self.get_maze_distance(agent_position, g.get_position()) for g in ghosts], default=float('inf'))
        reward += self.weights['ghost-distance'] * (1 / (ghost_dist + 1))  # Reward for avoiding ghosts

        return reward

    def final(self, state):
        """
        Called at the end of each game.
        """
        super().final(state)

    def get_successor(self, game_state, action):
        """
        Finds the next successor state resulting from the given action.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def compute_value_from_q_values(self, game_state):
        """
        Returns the max Q-value for the state.
        """
        allowed_actions = game_state.get_legal_actions(self.index)
        if len(allowed_actions) == 0:
            return 0.0
        best_action = self.get_policy(game_state)
        return self.get_q_value(game_state, best_action)

    def compute_action_from_q_values(self, game_state):
        """
        Computes the best action from Q-values.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if len(legal_actions) == 0:
            return None
        action_vals = {}
        best_q_value = float('-inf')
        for action in legal_actions:
            target_q_value = self.get_q_value(game_state, action)
            action_vals[action] = target_q_value
            if target_q_value > best_q_value:
                best_q_value = target_q_value
        best_actions = [k for k, v in action_vals.items() if v == best_q_value]
        return random.choice(best_actions)

    def get_policy(self, game_state):
        return self.compute_action_from_q_values(game_state)

    def get_value(self, game_state):
        return self.compute_value_from_q_values(game_state)

class FeaturesExtractor:

    def __init__(self, agentInstance):
        self.agentInstance = agentInstance

    def get_features(self, game_state, action):
        food = self.agentInstance.get_food(game_state)
        walls = game_state.get_walls()
        enemies = [game_state.get_agent_state(i) for i in self.agentInstance.get_opponents(game_state)]
        ghosts = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() is not None]

        features = util.Counter()

        features["bias"] = 1.0

        agent_position = game_state.get_agent_position(self.agentInstance.index)
        x, y = agent_position
        dx, dy = Actions.direction_to_vector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.get_legal_neighbors(g, walls) for g in ghosts)

        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = self.closest_food((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # Add feature for ghost distance
        ghost_dist = min([self.agentInstance.get_maze_distance((next_x, next_y), g) for g in ghosts], default=float('inf'))
        features["ghost-distance"] = 1 / (ghost_dist + 1)  # Reward for avoiding ghosts

        features.divide_all(10.0)
        return features

    def closest_food(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            if food[pos_x][pos_y]:
                return dist
            nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist + 1))
        return None

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}

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
    
class O(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def choose_action(self, game_state):
        # Get current agent state and position
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Check if the agent is carrying food (Pacman form)
        carrying_food = my_state.num_carrying > 0 

        if carrying_food:
            # If carrying food, try to return it to the starting position (home side)
            return self.return_food(game_state)
        else:
            # If not carrying food, go for food in enemy territory
            return self.collect_food(game_state)

    def return_food(self, game_state):
        """
        Returns the agent to its home side to deposit food.
        """
        # Get the legal actions and the current agent's state
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Determine which side of the board is the home side for this agent
        if self.red:
            home = game_state.get_red_food()  # Red team's food
        else:
            home = game_state.get_blue_food()  # Blue team's food
        
        # Convert the grid (home) to a list of positions where food is present
        home_positions = []
        for x in range(home.width):
            for y in range(home.height):
                if home[x][y]:  # If there's food at position (x, y)
                    home_positions.append((x, y))

        # If no home food found, return a random legal action
        if not home_positions:
            return random.choice(game_state.get_legal_actions(self.index))

        # Find the closest home position
        target_food = min(home_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
        
        # Move towards the closest food position
        return self.get_best_action(game_state, target_food)


    def collect_food(self, game_state):
        """
        Go to the nearest food in enemy territory.
        """
        # Get the food that we want to collect (returns a Grid)
        food = self.get_food(game_state)
        
        # Convert the Grid to a list of positions where food is present
        food_positions = []
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:  # If there is food at position (x, y)
                    food_positions.append((x, y))

        # If no food is found (edge case), return a random legal action
        if not food_positions:
            return random.choice(game_state.get_legal_actions(self.index))

        # Find the closest food position
        target_food = min(food_positions, key=lambda pos: self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), pos))
        
        # Move towards the food
        return self.get_best_action(game_state, target_food)


    
    def get_best_action(self, game_state, target):
        """
        Find the best legal action towards a given target position.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if 'Stop' in legal_actions:
            legal_actions.remove('Stop')
        
        # Find the best action based on distance to target
        best_action = None
        min_dist = float('inf')
        
        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(new_pos, target)
            if dist < min_dist:
                best_action = action
                min_dist = dist
        
        return best_action

class OffensiveAgent(CaptureAgent):
    """
    An offensive agent that collects food and returns home to deposit it.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        self.is_red = game_state.is_on_red_team(self.index)
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        self.mid_width = self.width // 2
        # Define the boundary based on team color
        if self.is_red:
            self.boundary = self.mid_width - 1
        else:
            self.boundary = self.mid_width
        # Precompute the home positions for returning
        self.home_positions = [
            (self.boundary, y)
            for y in range(self.height)
            if not self.walls[self.boundary][y]
        ]
        # List of legal positions for pathfinding
        self.legal_positions = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if not self.walls[x][y]
        ]

    def choose_action(self, game_state):
        """
        Uses expectimax search to choose the best action.
        """
        depth = 2  # Set the search depth
        action = self.expectimax(game_state, depth, self.index)[1]
        if action is None:
            # Fallback to a random legal action if expectimax fails
            legal_actions = game_state.get_legal_actions(self.index)
            if Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        return action

    def expectimax(self, game_state, depth, agent_index):
        """
        Performs expectimax search.
        """
        if depth == 0 or game_state.is_over():
            return self.evaluation_function(game_state), None

        # Check if the agent is alive (has a valid position)
        agent_state = game_state.get_agent_state(agent_index)
        agent_pos = agent_state.get_position()
        if agent_pos is None:
            # Agent is not active; skip to the next agent without changing depth
            next_agent_index = self.get_next_agent_index(game_state, agent_index)
            return self.expectimax(game_state, depth, next_agent_index)

        legal_actions = game_state.get_legal_actions(agent_index)
        if not legal_actions:
            # No legal actions available; skip to the next agent
            next_agent_index = self.get_next_agent_index(game_state, agent_index)
            return self.expectimax(game_state, depth, next_agent_index)

        if agent_index == self.index:
            # Max node (our agent)
            max_value = float('-inf')
            best_action = None
            for action in legal_actions:
                successor = game_state.generate_successor(agent_index, action)
                value = self.expectimax(
                    successor, depth - 1, self.get_next_agent_index(successor, agent_index)
                )[0]
                if value > max_value:
                    max_value = value
                    best_action = action
            return max_value, best_action
        else:
            # Expectation node (opponent)
            values = []
            for action in legal_actions:
                successor = game_state.generate_successor(agent_index, action)
                value = self.expectimax(
                    successor, depth, self.get_next_agent_index(successor, agent_index)
                )[0]
                values.append(value)
            average_value = sum(values) / len(values) if values else 0
            return average_value, None

    def get_next_agent_index(self, game_state, current_index):
        """
        Returns the next agent index, looping back to zero if necessary.
        """
        total_agents = game_state.get_num_agents()
        return (current_index + 1) % total_agents

    def evaluation_function(self, game_state):
        """
        Evaluation function considering various factors.
        """
        features = self.get_features(game_state)
        weights = self.get_weights(game_state)
        return features * weights

    def get_features(self, game_state):
        features = util.Counter()
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Distance to the closest food
        food_list = self.get_food(game_state).as_list()
        if food_list:
            min_food_distance = min(
                [self.get_maze_distance(my_pos, food) for food in food_list]
            )
            features['distance_to_food'] = float(min_food_distance) / (self.width * self.height)
        else:
            features['distance_to_food'] = 0.0

        # Number of food carried
        features['carrying'] = my_state.num_carrying

        # Check if agent should return home
        if my_state.is_pacman and my_state.num_carrying > 0:
            features['should_return_home'] = 1
            # Compute distance to home
            min_home_distance = min(
                [self.get_maze_distance(my_pos, pos) for pos in self.home_positions]
            )
            features['distance_to_home'] = float(min_home_distance) / (self.width * self.height)
        else:
            features['should_return_home'] = 0
            features['distance_to_home'] = 0.0

        # Avoid enemy ghosts
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [
            a
            for a in enemies
            if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0
        ]
        if ghosts:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in ghosts]
            min_ghost_distance = min(dists)
            features['distance_to_ghost'] = float(min_ghost_distance) / (self.width * self.height)
            if min_ghost_distance <= 1:
                features['ghost_nearby'] = 1
            else:
                features['ghost_nearby'] = 0
        else:
            features['distance_to_ghost'] = 1.0
            features['ghost_nearby'] = 0

        # Power capsules
        capsules = self.get_capsules(game_state)
        if capsules:
            min_capsule_distance = min(
                [self.get_maze_distance(my_pos, cap) for cap in capsules]
            )
            features['distance_to_capsule'] = float(min_capsule_distance) / (self.width * self.height)
        else:
            features['distance_to_capsule'] = 0.0

        return features

    def get_weights(self, game_state):
        """
        Returns weights for the features.
        """
        my_state = game_state.get_agent_state(self.index)
        weights = {
            'distance_to_food': -100.0,
            'carrying': 0.0,
            'should_return_home': 0.0,
            'distance_to_home': 0.0,
            'distance_to_ghost': 500.0,
            'ghost_nearby': -10000.0,
            'distance_to_capsule': -20.0,
        }

        if my_state.is_pacman and my_state.num_carrying > 0:
            # When carrying food, prioritize returning home
            weights['distance_to_food'] = 0.0
            weights['should_return_home'] = 10000.0
            weights['distance_to_home'] = -1000.0
            weights['distance_to_ghost'] = 1000.0
            weights['ghost_nearby'] = -20000.0
        else:
            # When not carrying food, prioritize collecting food
            weights['distance_to_food'] = -100.0
            weights['should_return_home'] = 0.0
            weights['distance_to_home'] = 0.0

        return weights

    def get_maze_distance(self, pos1, pos2):
        """
        Returns the maze distance between two positions.
        """
        return self.distancer.get_distance(pos1, pos2)

class PColomer(CaptureAgent):

    def register_initial_state(self, game_state):
        """
        Initialize the agent's internal state.
        """
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        """
        Choose the best action by evaluating successor positions.
        Switches between offensive and defensive modes based on the game score.
        """
        # Check if we are winning
        if self.get_score(game_state) > 0:
            # If winning, play defensively
            return self.choose_defensive_action(game_state)
        else:
            # If not winning, play offensively
            return self.choose_offensive_action(game_state)

    def choose_offensive_action(self, game_state):
        """
        Offensive behavior (similar to previous implementation).
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)  # Avoid stopping if possible

        best_action = None
        best_value = float('-inf')

        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_position(self.index)
            value = self.evaluate_offensive_position(successor, successor_pos)

            if value > best_value:
                best_value = value
                best_action = action
            elif value == best_value:
                # Tie-breaker: randomly choose between equally good actions
                best_action = random.choice([best_action, action])

        # Fallback to random action if no best action is found
        if best_action is None:
            print("No best action found. Choosing random action.")
            best_action = random.choice(legal_actions) if legal_actions else Directions.STOP

        return best_action

    def choose_defensive_action(self, game_state):
        """
        Defensive behavior as per DefensiveReflexAgent.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)  # Avoid stopping if possible

        best_action = None
        best_value = float('-inf')

        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            features = self.get_defensive_features(game_state, action)
            weights = self.get_defensive_weights(game_state, action)
            value = features * weights

            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            best_action = Directions.STOP

        return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor state resulting from the given action.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_position(self.index)
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate_offensive_position(self, game_state, position):
        """
        Evaluate a position based on offensive features and weights.
        """
        features = {}
        weights = {}

        agent_state = game_state.get_agent_state(self.index)
        food_list = self.get_food(game_state).as_list()
        ghost_positions = self.get_ghosts_positions(game_state)
        carrying_food = agent_state.num_carrying

        # Compute the distance to the closest food
        if food_list:
            min_food_distance = min(self.get_maze_distance(position, food_pos) for food_pos in food_list)
            features['distance_to_food'] = -min_food_distance  # Inverted to prioritize closer food
        else:
            features['distance_to_food'] = 0

        # Compute the distance to the closest ghost
        if ghost_positions:
            min_ghost_distance = min(self.get_maze_distance(position, ghost_pos) for ghost_pos in ghost_positions)
            features['distance_to_ghost'] = -min_ghost_distance  # Inverted to avoid ghosts
        else:
            features['distance_to_ghost'] = 0  # No ghosts visible

        # Compute the distance to home
        home_distance = self.get_distance_to_home(game_state, position)
        features['distance_to_home'] = -home_distance  # Inverted to prioritize returning home

        # Whether the position is on the enemy side
        is_on_enemy_side = self.is_on_enemy_side(position, game_state)
        features['on_enemy_side'] = 1 if is_on_enemy_side else 0

        # Penalty for carrying food
        features['carrying_food'] = carrying_food

        # Set the weights
        # Base weights
        weights['distance_to_food'] = 100  # Always prioritize food
        weights['distance_to_ghost'] = -5  # Avoid ghosts
        weights['distance_to_home'] = 0    # Base weight, will adjust below
        weights['on_enemy_side'] = 0       # Base weight, will adjust below
        weights['carrying_food'] = 0       # Base weight, will adjust below

        if carrying_food > 0:
            # Increase priority to return home based on food carried
            weights['distance_to_home'] = 100 * carrying_food
            # Increase penalty for being on enemy side when carrying food
            weights['on_enemy_side'] = -100 * carrying_food
            # Penalty for carrying food too long
            weights['carrying_food'] = -10 * carrying_food
            # Strongly avoid ghosts when carrying food
            weights['distance_to_ghost'] = -100 * carrying_food
            # Deprioritize collecting more food
            weights['distance_to_food'] = 0
        else:
            # Encourage being on enemy side to collect food
            weights['on_enemy_side'] = 50
            # Reset weights related to returning home
            weights['distance_to_home'] = 0
            weights['carrying_food'] = 0

        # Compute the evaluation score
        evaluation = sum(features[feature] * weights.get(feature, 0) for feature in features)

        return evaluation

    # Defensive features and weights
    def get_defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_defensive_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }

    # Additional methods needed from previous code
    def get_distance_to_home(self, game_state, position):
        """
        Calculates the shortest maze distance from a position to the home boundary.
        """
        home_boundary = self.get_home_boundary(game_state)
        if not home_boundary:
            return float('inf')
        return min(self.get_maze_distance(position, pos) for pos in home_boundary)

    def get_home_boundary(self, game_state):
        """
        Returns a list of positions on the agent's home boundary.
        """
        layout = game_state.data.layout
        mid_x = (layout.width // 2) - 1 if self.red else (layout.width // 2)
        home_boundary = [(mid_x, y) for y in range(layout.height) if not game_state.has_wall(mid_x, y)]
        return home_boundary

    def is_on_enemy_side(self, position, game_state):
        """
        Determines if the given position is on the enemy's side.
        """
        mid_x = game_state.data.layout.width // 2
        if self.red:
            return position[0] >= mid_x
        else:
            return position[0] < mid_x

    def get_ghosts_positions(self, game_state):
        """
        Returns the positions of all ghosts on the enemy team.
        """
        enemies = self.get_opponents(game_state)
        return [
            game_state.get_agent_position(enemy)
            for enemy in enemies
            if not game_state.get_agent_state(enemy).is_pacman and game_state.get_agent_position(enemy) is not None
        ]

#versió sense la defensa millorada
class WinnerAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        """
        Initialize the agent's internal state.
        """
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        """
        Choose the best action by evaluating successor positions.
        Switches between offensive and defensive modes based on the game score.
        """
        # Check if we are winning
        if self.get_score(game_state) > 3:
            # If winning, play defensively
            return self.choose_defensive_action(game_state)
        else:
            # If not winning, play offensively
            return self.choose_offensive_action(game_state)

    def choose_offensive_action(self, game_state):
        """
        Offensive behavior using methods from the provided class O.
        """
        # Get current agent state and position
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Check if the agent is carrying food
        carrying_food = my_state.num_carrying > 0

        if carrying_food:
            # If carrying food, try to return to home
            return self.return_food(game_state)
        else:
            # If not carrying food, go for food in enemy territory
            return self.collect_food(game_state)

    def return_food(self, game_state):
        """
        Returns the agent to its home side to deposit food.
        """
        my_pos = game_state.get_agent_position(self.index)

        # Get the home boundary positions
        home_boundary = self.get_home_boundary(game_state)

        # If no home boundary positions found, return a random legal action
        if not home_boundary:
            return random.choice(game_state.get_legal_actions(self.index))

        # Find the closest home boundary position
        target_pos = min(home_boundary, key=lambda pos: self.get_maze_distance(my_pos, pos))

        # Move towards the target position
        return self.get_best_action(game_state, target_pos)

    def collect_food(self, game_state):
        """
        Go to the nearest food in enemy territory.
        """
        # Get the food that we want to collect (returns a Grid)
        food = self.get_food(game_state)
        
        # Convert the Grid to a list of positions where food is present
        food_positions = []
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:  # If there is food at position (x, y)
                    food_positions.append((x, y))

        # If no food is found (edge case), return a random legal action
        if not food_positions:
            return random.choice(game_state.get_legal_actions(self.index))

        # Find the closest food position
        my_pos = game_state.get_agent_state(self.index).get_position()
        target_food = min(food_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))
        
        # Move towards the food
        return self.get_best_action(game_state, target_food)

    def get_best_action(self, game_state, target):
        """
        Find the best legal action towards a given target position.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)  # Avoid stopping if possible

        # Find the best action based on distance to target
        best_action = None
        min_dist = float('inf')

        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist = self.get_maze_distance(new_pos, target)
            if dist < min_dist:
                best_action = action
                min_dist = dist
            elif dist == min_dist:
                # Tie-breaker
                best_action = random.choice([best_action, action])

        return best_action

    def get_home_boundary(self, game_state):
        """
        Returns a list of positions on the agent's home boundary.
        """
        layout = game_state.data.layout
        mid_x = (layout.width // 2) - 1 if self.red else (layout.width // 2)
        home_boundary = [(mid_x, y) for y in range(layout.height)
                         if not game_state.has_wall(mid_x, y)]
        return home_boundary

    def choose_defensive_action(self, game_state):
        """
        Defensive behavior as per DefensiveReflexAgent.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)  # Avoid stopping if possible

        best_action = None
        best_value = float('-inf')

        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            features = self.get_defensive_features(game_state, action)
            weights = self.get_defensive_weights(game_state, action)
            value = features * weights

            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            best_action = Directions.STOP

        return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor state resulting from the given action.
        """
        successor = game_state.generate_successor(self.index, action)
        return successor

    # Defensive features and weights
    def get_defensive_features(self, game_state, action):
        features = Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            features['invader_distance'] = 0

        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def get_defensive_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2
        }