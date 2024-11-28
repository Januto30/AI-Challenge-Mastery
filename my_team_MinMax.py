import random
import util
import time

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point



#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
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
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.memory = {}  # For remembering positions (e.g., enemy positions)
        
    def update_memory(self, game_state):
        """Update memory with recent positions of invaders or food."""
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        
        for invader in invaders:
            self.memory[invader.get_position()] = invader


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def is_terminal_state(self, game_state, depth):
        """
        Checks if the state is terminal (game over or depth limit reached).
        """
        if game_state.is_over() or depth == 0:
            return True
        return False

    def choose_action(self, game_state):
        """Optimized Minimax with Alpha-Beta and time management."""
        start_time = time.time()
        alpha = float('-inf')
        beta = float('inf')

        # Initialize best action and value
        best_action = None
        best_value = float('-inf')

        max_depth = 3  # Start with shallow depth
        while time.time() - start_time < self.time_for_computing:
            try:
                for action in game_state.get_legal_actions(self.index):
                    successor = self.get_successor(game_state, action)
                    value = self.min_value(successor, max_depth - 1, alpha, beta)
                    if value > best_value:
                        best_value = value
                        best_action = action
                    alpha = max(alpha, best_value)
            except TimeoutError:
                break  # Stop searching if we run out of time
            max_depth += 1  # Increase depth for iterative deepening

        return best_action


    def get_best_action_towards_target(self, game_state, target_position):
        """
        Chooses the best action that moves the agent towards the target position.
        This method calculates the best action by checking all legal actions and seeing
        which one brings the agent closest to the target.
        """
        best_action = None
        min_distance = float('inf')

        # Try each possible action and see which one gets us closer to the target
        actions = game_state.get_legal_actions(self.index)
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist_to_target = self.get_maze_distance(new_pos, target_position)

            if dist_to_target < min_distance:
                min_distance = dist_to_target
                best_action = action

        return best_action

    def min_value(self, game_state, depth, alpha, beta):
        """Min player for minimax (handles opponents' moves)."""
        if self.is_terminal_state(game_state, depth):
            return self.evaluate_state(game_state)

        value = float('inf')
        actions = game_state.get_legal_actions(self.index)
        for action in actions:
            successor = self.get_successor(game_state, action)
            value = min(value, self.max_value(successor, depth - 1, alpha, beta))
            if value < alpha:
                return value
            beta = min(beta, value)

        return value

    def max_value(self, game_state, depth, alpha, beta):
        """Max player for minimax (handles our agent's moves)."""
        if self.is_terminal_state(game_state, depth):
            return self.evaluate_state(game_state)

        value = float('-inf')
        actions = game_state.get_legal_actions(self.index)
        for action in actions:
            successor = self.get_successor(game_state, action)
            value = max(value, self.min_value(successor, depth - 1, alpha, beta))
            if value > beta:
                return value
            alpha = max(alpha, value)

        return value

    def evaluate_state(self, game_state):
        """
        Evaluate the game state for use in minimax, reflecting more dynamic features
        to prevent agents from getting stuck.
        """
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Penalize being stationary (avoid getting stuck)
        if my_pos == self.start:
            return -float('inf')  # Penalize if the agent is stuck at the starting position

        # Initialize dynamic evaluation features
        features = util.Counter()

        # Offensive agent: prioritize food collection
        if isinstance(self, OffensiveReflexAgent):
            food_list = self.get_food(game_state).as_list()
            if len(food_list) > 0:
                nearest_food = min([self.get_maze_distance(my_pos, food) for food in food_list])
                features['distance_to_food'] = nearest_food
            else:
                features['distance_to_food'] = 0  # No food left

        # Defensive agent: prioritize invader defense
        elif isinstance(self, DefensiveReflexAgent):
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

            if len(invaders) > 0:
                # If invaders are close, prioritize getting to them
                dist_to_invader = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders])
                features['invader_distance'] = dist_to_invader
            else:
                # Patrol around the defensive zone if no invaders are near
                patrol_points = self.get_patrol_points(game_state)
                if patrol_points:
                    dist_to_patrol = min([self.get_maze_distance(my_pos, pt) for pt in patrol_points])
                    features['patrol_distance'] = dist_to_patrol

        # Distance to home base for defensive agents (for patrolling or retreating)
        if isinstance(self, DefensiveReflexAgent):
            home_base = game_state.get_agent_state(self.index).get_position()  # Assuming home base is tracked.
            features['distance_to_home'] = self.get_maze_distance(my_pos, home_base)
        
        # Example: Penalizing 'stop' and reversing directions to avoid non-optimal behavior
        features['stop'] = 1 if game_state.get_agent_state(self.index).configuration.direction == Directions.STOP else 0
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        reverse_direction = Directions.REVERSE[current_direction]
        if game_state.get_agent_state(self.index).configuration.direction == reverse_direction:
            features['reverse'] = 1

        # Feature weights (customizable based on the agent's behavior)
        weights = {
            'distance_to_food': -1,
            'invader_distance': -5,
            'patrol_distance': -2,
            'distance_to_home': -2,  # Encourage going home if no invaders are near
            'stop': -100,            # Discourage stopping
            'reverse': -2            # Discourage reversing
        }

        return features * weights

    def get_successor(self, game_state, action):
        """Find the next successor which is a grid position (location tuple)."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor


class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.memory = {}  # For remembering positions (e.g., enemy positions)

    def evaluate_state(self, game_state):
        features = util.Counter()
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        # Distance to food
        food_list = self.get_food(game_state).as_list()
        features['distance_to_food'] = min(
            [self.get_maze_distance(my_pos, food) for food in food_list], default=0
        )

        # Distance to power pellets
        capsules = self.get_capsules(game_state)
        features['distance_to_capsule'] = min(
            [self.get_maze_distance(my_pos, cap) for cap in capsules], default=float('inf')
        )

        # Penalize proximity to enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_enemies = [e for e in enemies if e.get_position() and not e.scared_timer]
        if visible_enemies:
            nearest_enemy_dist = min(
                [self.get_maze_distance(my_pos, e.get_position()) for e in visible_enemies]
            )
            features['distance_to_enemy'] = nearest_enemy_dist
            features['near_enemy'] = 1 if nearest_enemy_dist < 5 else 0

        # Penalize stopping and reversing
        features['stop'] = 1 if game_state.get_agent_state(self.index).configuration.direction == Directions.STOP else 0
        features['reverse'] = 1 if game_state.get_agent_state(self.index).configuration.direction == Directions.REVERSE[
            game_state.get_agent_state(self.index).configuration.direction
        ] else 0

        weights = {
            'distance_to_food': -1,
            'distance_to_capsule': -2,
            'distance_to_enemy': 2,
            'near_enemy': -500,
            'stop': -100,
            'reverse': -2,
        }
        return features * weights

    
    def choose_action(self, game_state):
        start_time = time.time()

        # Get the list of food and the agent's position
        food_list = self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        # Get the agent's state, including scared timer
        my_state = game_state.get_agent_state(self.index)
        scared_timer = my_state.scared_timer  # Directly access the scared_timer here

        # Check for nearby enemies (non-scared)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        phantoms = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # If the agent is scared, prioritize avoiding enemies
        if scared_timer > 0:
            # Avoid enemies if scared
            nearest_phantom_distance = float('inf')
            nearest_phantom_pos = None
            
            if phantoms:
                for phantom in phantoms:
                    phantom_pos = phantom.get_position()
                    distance_to_phantom = self.get_maze_distance(my_pos, phantom_pos)
                    if distance_to_phantom < nearest_phantom_distance:
                        nearest_phantom_distance = distance_to_phantom
                        nearest_phantom_pos = phantom_pos

            # If phantoms are close, try to avoid them by moving away
            if nearest_phantom_pos and nearest_phantom_distance < 5:  # Threshold for "close" can be adjusted
                best_action = self.get_best_action_away_from_target(game_state, nearest_phantom_pos)
                return best_action

        # Otherwise, if not scared, prioritize food collection
        if len(food_list) > 0:
            nearest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))
            best_action = self.get_best_action_towards_target(game_state, nearest_food)
            return best_action

        # If no food is nearby, fallback to minimax (or other behaviors)
        alpha = float('-inf')
        beta = float('inf')

        max_time = 0.1  # Max time limit for computation
        best_action = None
        best_value = float('-inf')

        for depth in range(1, 4):  # Iterative deepening
            if time.time() - start_time > max_time:
                break

            actions = game_state.get_legal_actions(self.index)

            for action in actions:
                successor = self.get_successor(game_state, action)
                value = self.min_value(successor, depth - 1, alpha, beta)

                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)

        return best_action

    def get_best_action_away_from_target(self, game_state, target_position):
        """
        Chooses the best action that moves the agent away from the target position (e.g., a phantom).
        """
        best_action = None
        max_distance = -float('inf')

        # Try each possible action and see which one moves us furthest from the target (phantom)
        actions = game_state.get_legal_actions(self.index)
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist_to_target = self.get_maze_distance(new_pos, target_position)

            if dist_to_target > max_distance:  # Maximize distance from the phantom
                max_distance = dist_to_target
                best_action = action

        return best_action
    
class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that uses minimax with alpha-beta pruning to play defensively.
    It actively patrols its territory and dynamically reacts to invaders.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.patrol_points = None  # Will store points for patrolling
        self.last_patrol_point = None  # Track last patrol point to vary patrol behavior
        self.patrol_target = None  # Keeps track of the current patrol target

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.patrol_points = self.get_patrol_points(game_state)

    def get_patrol_points(self, game_state):
        """
        Generate patrol points within the defensive zone.
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height

        # Generate points in the middle of the agent's defensive half
        mid_x = width // 2
        if self.red:
            patrol_x = range(1, mid_x)  # Red team's side
        else:
            patrol_x = range(mid_x, width - 1)  # Blue team's side

        patrol_points = [(x, y) for x in patrol_x for y in range(1, height - 1) if not walls[x][y]]
        return patrol_points

    def evaluate_state(self, game_state):
        features = util.Counter()
        my_pos = game_state.get_agent_state(self.index).get_position()
        
        # Invader features
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            features['invader_distance'] = min(dists)

        # Patrol features
        if not invaders:
            patrol_points = self.get_patrol_points(game_state)
            patrol_dists = [self.get_maze_distance(my_pos, point) for point in patrol_points]
            features['patrol_distance'] = min(patrol_dists, default=float('inf'))

        weights = {
            'num_invaders': -1000,
            'invader_distance': -10,
            'patrol_distance': -2,
        }
        return features * weights

    def choose_action(self, game_state):
        """
        Use alpha-beta minimax to decide on the best defensive action.
        """
        actions = game_state.get_legal_actions(self.index)

        # Check if there are invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]

        # If invaders are detected, the agent should ignore patrol points and focus on the invader
        if invaders:
            # Find the nearest invader
            my_pos = game_state.get_agent_state(self.index).get_position()
            invader_positions = [invader.get_position() for invader in invaders]
            closest_invader_pos = min(invader_positions, key=lambda pos: self.get_maze_distance(my_pos, pos))

            # Move towards the closest invader
            best_action = self.get_best_action_towards_target(game_state, closest_invader_pos)
        else:
            # If no invaders, patrol. Select the next patrol target if the agent is too close to the current patrol point
            if not self.patrol_target or self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), self.patrol_target) < 2:
                self.patrol_target = self.select_new_patrol_point()

            best_action = self.get_best_action_towards_target(game_state, self.patrol_target)

        return best_action

    def select_new_patrol_point(self):
        """
        Selects a new patrol point at random, ensuring it's not the same as the last one.
        """
        # Choose a new patrol point that's not too close to the last one
        new_patrol_point = random.choice(self.patrol_points)
        while new_patrol_point == self.patrol_target:
            new_patrol_point = random.choice(self.patrol_points)
        return new_patrol_point

    def get_best_action_towards_target(self, game_state, target_position):
        """
        Chooses the best action that moves the agent towards the target position.
        """
        best_action = None
        min_distance = float('inf')

        # Try each possible action and see which one gets us closer to the target
        actions = game_state.get_legal_actions(self.index)
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            dist_to_target = self.get_maze_distance(new_pos, target_position)

            if dist_to_target < min_distance:
                min_distance = dist_to_target
                best_action = action

        return best_action
