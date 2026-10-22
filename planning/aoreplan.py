import shutil
import sys

import numpy as np
from pogema import GridConfig

# Use Python on Windows and the C++ planner elsewhere.
if sys.platform == "win32":
    from planning.planner_python import planner
else:
    import cppimport.import_hook

    if shutil.which("g++") is None:
        cppimport.settings["release_mode"] = True
    from planning.planner import planner

INF = 1_000_000_000
FAILURE_CACHE_PROBABILITY = 0.5
_FAILURE_CACHE_SEED_SALT = 0xFA11CA

_MOVES = tuple(tuple(move) for move in GridConfig().MOVES)
_ACTION_BY_DELTA = {move: index for index, move in enumerate(_MOVES)}


def _update_map(local_planner, observation, other_agents):
    # Merge local obstacles and visible agents into the planner's map.
    obstacles = np.asarray(observation["obstacles"])
    origin = tuple(int(p) - size // 2 for p, size in zip(observation["xy"], obstacles.shape))
    local_planner.update_obstacles(np.argwhere(obstacles), other_agents, origin)


def _path_action(path):
    # Convert the first path edge to an action, or return None if unavailable.
    if path is None or path[1][0] >= INF:
        return None
    start, destination = path
    return _ACTION_BY_DELTA[(destination[0] - start[0], destination[1] - start[1])]



class AORePlanBase:
    def __init__(self, max_steps: int = INF, seed=None):
        self.planner = None
        self.max_steps = int(max_steps)
        self.rnd = np.random.default_rng(seed)
        cache_seed = None if seed is None else np.random.SeedSequence(
            [int(seed), _FAILURE_CACHE_SEED_SALT]
        )
        self.failure_cache_rnd = np.random.default_rng(cache_seed)

    def act(self, observations, skip_agents=None):
        # Produce one dynamic-planner proposal per agent.
        count = len(observations)
        if skip_agents is None:
            skip = [False] * count
        else:
            skip = skip_agents
        if len(skip) != count:
            raise ValueError("skip_agents and observations must have equal sizes.")

        if self.planner is None:
            self.planner = [planner(self.max_steps) for _ in range(count)]
        elif len(self.planner) != count:
            raise ValueError("Reset AORePlan before changing the agent count.")

        actions = []
        for index, observation in enumerate(observations):
            position = tuple(map(int, observation["xy"]))
            target = tuple(map(int, observation["target_xy"]))
            local_planner = self.planner[index]
            _update_map(local_planner, observation, np.argwhere(observation["agents"]))

            cache_failed_action = True
            if local_planner.proposal_failed(position):
                # Cache failed moves with probability 0.5 to break repeated conflicts.
                cache_failed_action = self.failure_cache_rnd.random() < FAILURE_CACHE_PROBABILITY
            local_planner.observe_position(position, cache_failed_action)

            if position == target or skip[index]:
                actions.append(None)
                continue

            local_planner.plan_path(position, target)
            action = _path_action(local_planner.get_next_node(True))
            if action is None:
                # Release failed moves when the planner cannot propose an action.
                local_planner.release_failed_actions()
            actions.append(action)
        return actions

    def commit_proposals(self, executed_mask):
        # Cancel unused proposals so they are not recorded as failed moves.
        if self.planner is None:
            raise RuntimeError("AORePlan must act before proposals can be committed.")
        if len(executed_mask) != len(self.planner):
            raise ValueError("executed_mask and AORePlan planners must have equal sizes.")
        for local_planner, executed in zip(self.planner, executed_mask):
            if not executed:
                local_planner.cancel_desired()


def _local_cell_is_free(observation, action):
    # Check the destination against currently visible obstacles and agents.
    if action in (None, 0):
        return False
    try:
        obstacles = np.asarray(observation["obstacles"])
        agents = np.asarray(observation["agents"])
        di, dj = _MOVES[int(action)]
        i, j = obstacles.shape[0] // 2 + di, obstacles.shape[1] // 2 + dj
        return bool(
            0 <= i < obstacles.shape[0]
            and 0 <= j < obstacles.shape[1]
            and obstacles[i, j] == 0
            and agents[i, j] == 0
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def original_random_or_stay(observation, rnd):
    # Choose wait or a random move; random moves only avoid static obstacles.
    if rnd.random() <= 0.5:
        return 0

    actions = [1, 2, 3, 4]
    rnd.shuffle(actions)
    obstacles = np.asarray(observation["obstacles"])
    i, j = obstacles.shape[0] // 2, obstacles.shape[1] // 2
    for action in actions:
        di, dj = _MOVES[action]
        if obstacles[i + di, j + dj] == 0:
            return action
    return 0


class StaticAStarCheck:
    def __init__(self, max_steps: int = INF):
        self.max_steps = int(max_steps)
        self._planners = None

    def observe(self, observations):
        # Accumulate static obstacles without adding other agents.
        count = len(observations)
        if self._planners is None:
            self._planners = [planner(self.max_steps) for _ in range(count)]
        elif len(self._planners) != count:
            raise ValueError("Static A* agent count changed without reset.")
        for local_planner, observation in zip(self._planners, observations):
            _update_map(local_planner, observation, [])

    def get_action(self, index, observation):
        # Return the first static-map action for the selected agent.
        if self._planners is None:
            raise RuntimeError("Observe the agent batch before querying static A*.")
        local_planner = self._planners[index]
        position = tuple(map(int, observation["xy"]))
        target = tuple(map(int, observation["target_xy"]))
        local_planner.update_static_path(position, target)
        action = _path_action(local_planner.get_next_node(False))
        local_planner.cancel_desired()
        return action


class AORePlanWrapper:
    def __init__(self, agent, max_steps: int = INF):
        self.agent = agent
        self.rnd = agent.rnd
        self.static_astar = StaticAStarCheck(max_steps=max_steps)
        self.previous_position = None
        self.last_target = None
        self.last_dynamic_override_mask = None

    def _ensure_state(self, count):
        if self.previous_position is None:
            self.previous_position = [None] * count
            self.last_target = [None] * count
            return
        if len(self.previous_position) != count:
            raise ValueError("AORePlan agent count changed without reset.")

    def _static_astar_action(self, index, observation):
        # Turn a locally blocked static-map action into wait.
        action = self.static_astar.get_action(index, observation)
        return int(action) if _local_cell_is_free(observation, action) else 0

    @staticmethod
    def _returns_to_previous_position(position, action, previous):
        # A reverse move returns to the position from the previous timestep.
        if action in (None, 0) or previous is None:
            return False
        dx, dy = _MOVES[int(action)]
        return (position[0] + dx, position[1] + dy) == previous

    def commit_proposals(self, executed_mask):
        self.agent.commit_proposals(
            [
                bool(executed) and not overridden
                for executed, overridden in zip(executed_mask, self.last_dynamic_override_mask)
            ]
        )

    def act(self, observations, skip_agents=None):
        actions = list(self.agent.act(observations, skip_agents=skip_agents))
        self._ensure_state(len(actions))

        self.static_astar.observe(observations)
        self.last_dynamic_override_mask = [False] * len(actions)

        for index, observation in enumerate(observations):
            raw_action = actions[index]
            position = tuple(map(int, observation["xy"]))
            target = tuple(map(int, observation["target_xy"]))
            if self.last_target[index] != target:
                self.previous_position[index] = None
                self.last_target[index] = target

            previous = self.previous_position[index]
            # Record every timestep, including waits and blocked moves.
            self.previous_position[index] = position

            if skip_agents is not None and bool(skip_agents[index]):
                continue

            if position == target:
                actions[index] = 0
                continue

            if raw_action is None:
                actions[index] = original_random_or_stay(observation, self.rnd)
                self.last_dynamic_override_mask[index] = True
                continue

            if not self._returns_to_previous_position(position, raw_action, previous):
                continue

            static_action = self._static_astar_action(index, observation)
            # Replace the reverse proposal only if the static action is not reverse.
            if not self._returns_to_previous_position(position, static_action, previous):
                actions[index] = int(static_action)
                self.last_dynamic_override_mask[index] = True

        return actions
