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
from util import Counter


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='TwoWayAgent', second='DefensiveReflexAgentImproved', num_training=0):
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


class DefensiveReflexAgentImproved(CaptureAgent):
    """
    An improved defensive agent that tracks invaders, patrols key positions,
    and avoids getting eaten when scared.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.patrol_points = self.get_patrol_points(game_state)
        self.target = None
        self.last_observed_food = None

    def choose_action(self, game_state):
        """
        Picks among actions that minimize the distance to invaders,
        or moves to patrol points if no invaders are detected.
        """
        # Update the list of food we're defending
        self.observe_food(game_state)

        # Compute the action values
        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP]  # Avoid stopping

        values = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            value = self.evaluate(game_state, successor)
            values.append((value, action))

        # Choose the best action
        best_value, best_action = max(values)

        return best_action

    def evaluate(self, game_state, successor):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, successor)
        weights = self.get_weights(game_state, successor)
        return features * weights

    def get_features(self, game_state, successor):
        features = util.Counter()

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Feature: whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Feature: number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Feature: distance to the closest invader
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            self.target = None  # Reset target since we see an invader
        else:
            # Infer the position of invaders based on missing food
            invader_positions = self.infer_invader_positions(game_state)
            if invader_positions:
                # Move towards the inferred position
                dists = [self.get_maze_distance(my_pos, pos) for pos in invader_positions]
                features['inferred_invader_distance'] = min(dists)
                self.target = invader_positions[0]
            else:
                # No invaders detected; patrol
                if self.target is None or my_pos == self.target:
                    self.target = random.choice(self.patrol_points)
                features['patrol_distance'] = self.get_maze_distance(my_pos, self.target)

        # Feature: whether the agent is scared
        if my_state.scared_timer > 0:
            features['is_scared'] = 1
        else:
            features['is_scared'] = 0

        # Feature: distance to the closest capsule (if scared)
        if features['is_scared']:
            capsules = self.get_capsules_you_are_defending(game_state)
            if capsules:
                dists = [self.get_maze_distance(my_pos, cap) for cap in capsules]
                features['capsule_distance'] = min(dists)
            else:
                features['capsule_distance'] = 0
        else:
            features['capsule_distance'] = 0

        return features

    def get_weights(self, game_state, successor):
        weights = {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'inferred_invader_distance': -8,
            'patrol_distance': -1,
            'is_scared': -100,
            'capsule_distance': 10,
        }
        return weights

    def infer_invader_positions(self, game_state):
        """
        Infer the positions of invaders based on missing food.
        """
        current_food = self.get_food_you_are_defending(game_state).as_list()

        if self.last_observed_food is not None:
            missing_food = set(self.last_observed_food) - set(current_food)
            if missing_food:
                # Possible invader positions
                return list(missing_food)

        self.last_observed_food = current_food
        return []

    def observe_food(self, game_state):
        """
        Update the food we're defending.
        """
        self.last_observed_food = self.get_food_you_are_defending(game_state).as_list()

    def get_patrol_points(self, game_state):
        """
        Get points to patrol along the home boundary.
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        mid_x = (width // 2) - 1 if self.red else (width // 2)
        patrol_points = []

        for y in range(1, height - 1):
            if not walls[mid_x][y]:
                patrol_points.append((mid_x, y))

        # Reduce patrol points for efficiency
        patrol_points = patrol_points[::2]

        return patrol_points

    def get_successor(self, game_state, action):
        """
        Finds the next successor, which is a GameState object.
        """
        successor = game_state.generate_successor(self.index, action)
        return successor


class TwoWayAgent(CaptureAgent):

    def register_initial_state(self, game_state):
        """
        Initialize the agent's internal state.
        """
        super().register_initial_state(game_state)
        # For the improved defensive agent
        self.patrol_points = self.get_patrol_points(game_state)
        self.target = None
        self.last_observed_food = None

    def choose_action(self, game_state):
        """
        Choose the best action by evaluating successor positions.
        Switches between offensive and defensive modes based on the game score.
        """
        # Check if we are winning by more than 3 points
        if self.get_score(game_state) > 3:
            # If winning, play defensively
            return self.choose_defensive_action(game_state)
        else:
            # If not winning, play offensively
            return self.choose_offensive_action(game_state)

    def choose_offensive_action(self, game_state):
        """
        Offensive behavior.
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
        food_positions = food.as_list()

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
        Improved defensive behavior.
        """
        # Update the list of food we're defending
        self.observe_food(game_state)

        # Compute the action values
        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP]  # Avoid stopping

        values = []
        for action in actions:
            successor = self.get_successor(game_state, action)
            value = self.evaluate_defensive(game_state, successor)
            values.append((value, action))

        # Choose the best action
        best_value, best_action = max(values)

        return best_action

    def evaluate_defensive(self, game_state, successor):
        """
        Computes a linear combination of features and feature weights for defense.
        """
        features = self.get_defensive_features(game_state, successor)
        weights = self.get_defensive_weights(game_state, successor)
        return features * weights

    def get_defensive_features(self, game_state, successor):
        features = Counter()

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Feature: whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Feature: number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Feature: distance to the closest invader
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
            self.target = None  # Reset target since we see an invader
        else:
            # Infer the position of invaders based on missing food
            invader_positions = self.infer_invader_positions(game_state)
            if invader_positions:
                # Move towards the inferred position
                dists = [self.get_maze_distance(my_pos, pos) for pos in invader_positions]
                features['inferred_invader_distance'] = min(dists)
                self.target = invader_positions[0]
            else:
                # No invaders detected; patrol
                if self.target is None or my_pos == self.target:
                    self.target = random.choice(self.patrol_points)
                features['patrol_distance'] = self.get_maze_distance(my_pos, self.target)

        # Feature: whether the agent is scared
        if my_state.scared_timer > 0:
            features['is_scared'] = 1
        else:
            features['is_scared'] = 0

        # Feature: distance to the closest capsule (if scared)
        if features['is_scared']:
            capsules = self.get_capsules_you_are_defending(game_state)
            if capsules:
                dists = [self.get_maze_distance(my_pos, cap) for cap in capsules]
                features['capsule_distance'] = min(dists)
            else:
                features['capsule_distance'] = 0
        else:
            features['capsule_distance'] = 0

        return features

    def get_defensive_weights(self, game_state, successor):
        weights = {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'inferred_invader_distance': -8,
            'patrol_distance': -1,
            'is_scared': -100,
            'capsule_distance': 10,
        }
        return weights

    def get_successor(self, game_state, action):
        """
        Finds the next successor state resulting from the given action.
        """
        successor = game_state.generate_successor(self.index, action)
        return successor

    def infer_invader_positions(self, game_state):
        """
        Infer the positions of invaders based on missing food.
        """
        current_food = self.get_food_you_are_defending(game_state).as_list()

        if self.last_observed_food is not None:
            missing_food = set(self.last_observed_food) - set(current_food)
            if missing_food:
                # Possible invader positions
                return list(missing_food)

        self.last_observed_food = current_food
        return []

    def observe_food(self, game_state):
        """
        Update the food we're defending.
        """
        self.last_observed_food = self.get_food_you_are_defending(game_state).as_list()

    def get_patrol_points(self, game_state):
        """
        Get points to patrol along the home boundary.
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        mid_x = (width // 2) - 1 if self.red else (width // 2)
        patrol_points = []

        for y in range(1, height - 1):
            if not walls[mid_x][y]:
                patrol_points.append((mid_x, y))

        # Reduce patrol points for efficiency
        patrol_points = patrol_points[::2]

        return patrol_points

