import shutil
import sys

import numpy as np
from pogema import GridConfig

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
    # 输入：local_planner 为单个智能体的规划器；observation 提供障碍矩阵 obstacles 和位置 xy；
    #       other_agents 为其他智能体在局部矩阵中的格子索引，静态查询时传空列表。
    # 输出：无返回值；更新规划器内部的累计障碍地图和当前其他智能体位置。
    # 将局部观测中的格子索引转换为与 xy 一致的地图坐标。
    obstacles = np.asarray(observation["obstacles"])
    origin = tuple(int(p) - size // 2 for p, size in zip(observation["xy"], obstacles.shape))
    local_planner.update_obstacles(np.argwhere(obstacles), other_agents, origin)


def _path_action(path):
    # 输入：path 为（当前位置，下一格位置），也可能是 None 或包含无路径标记 INF。
    # 输出：对应的动作编号 0～4；没有可用下一格时返回 None。
    # 底层返回当前位置和路径的下一格；None 表示没有候选动作，不等于等待 0。
    if path is None or path[1][0] >= INF:
        return None
    start, destination = path
    return _ACTION_BY_DELTA[(destination[0] - start[0], destination[1] - start[1])]



class AORePlanBase:
    def __init__(self, max_steps: int = INF, seed=None):
        # 输入：max_steps 为单次搜索最多扩展的节点数，不是回合长度；seed 为随机种子。
        # 输出：无返回值；初始化搜索上限和两组随机数生成器，规划器在首次 act 时创建。
        self.planner = None
        self.max_steps = int(max_steps)
        self.rnd = np.random.default_rng(seed)
        # 缓存写入和随机后备动作使用独立随机流，避免相互改变抽样顺序。
        cache_seed = None if seed is None else np.random.SeedSequence(
            [int(seed), _FAILURE_CACHE_SEED_SALT]
        )
        self.failure_cache_rnd = np.random.default_rng(cache_seed)

    def act(self, observations, skip_agents=None):
        # 输入：observations 按智能体排列，每项包含 obstacles、agents 两张局部矩阵及 xy、target_xy；
        #       skip_agents 为可选的同长度布尔列表，True 表示该智能体本步不生成候选。
        # 输出：同长度候选动作列表；已到目标、被跳过或搜索失败的智能体对应 None。
        #       同时更新各智能体的地图、失败缓存和待核对的候选目标格。
        count = len(observations)
        if skip_agents is None:
            skip = [False] * count
        else:
            skip = skip_agents
        if len(skip) != count:
            raise ValueError("skip_agents and observations must have equal sizes.")

        if self.planner is None:
            # 每个智能体各自维护地图和失败缓存，不共享规划状态。
            self.planner = [planner(self.max_steps) for _ in range(count)]
        elif len(self.planner) != count:
            raise ValueError("Reset AORePlan before changing the agent count.")

        actions = []
        for index, observation in enumerate(observations):
            position = tuple(map(int, observation["xy"]))
            target = tuple(map(int, observation["target_xy"]))
            local_planner = self.planner[index]
            _update_map(local_planner, observation, np.argwhere(observation["agents"]))

            # 用实际位置核对上次提交的目标格，仅在移动失败时抽取缓存写入概率。
            # 缓存会让后续动态搜索暂时避开这些失败格；清理逻辑在搜索后端。
            cache_failed_action = True
            if local_planner.proposal_failed(position):
                cache_failed_action = self.failure_cache_rnd.random() < FAILURE_CACHE_PROBABILITY
            local_planner.observe_position(position, cache_failed_action)

            if position == target or skip[index]:
                # 即使不规划，也先更新地图和上一步执行反馈。
                actions.append(None)
                continue

            local_planner.plan_path(position, target)
            # True：优先返回完整路径首步；无完整路径时，退到 BestMove。
            # BestMove 是朝搜索过程中最接近目标的节点走的首步，不保证这一步靠近目标。
            action = _path_action(local_planner.get_next_node(True))
            if action is None:
                # 连候选动作也没有时释放失败缓存，避免缓存把后续搜索永久堵住。
                local_planner.release_failed_actions()
            actions.append(action)
        return actions

    def commit_proposals(self, executed_mask):
        # 输入：executed_mask 为同长度布尔列表，True 表示该动态候选已被提交给环境。
        # 输出：无返回值；取消未被提交的候选目标格记录，保留已提交候选的记录。
        # 这里确认的是候选是否被提交，不是是否移动成功；成功与否由下一次观测判断。
        if self.planner is None:
            raise RuntimeError("AORePlan must act before proposals can be committed.")
        if len(executed_mask) != len(self.planner):
            raise ValueError("executed_mask and AORePlan planners must have equal sizes.")
        for local_planner, executed in zip(self.planner, executed_mask):
            if not executed:
                # 未采用的候选不能在下一步被误记为一次移动失败。
                local_planner.cancel_desired()


def _local_cell_is_free(observation, action):
    # 输入：observation 提供 obstacles、agents 局部矩阵；action 为待检查的动作编号。
    # 输出：移动目标格在观测范围内且当前无障碍、无智能体时返回 True，否则返回 False；
    #       等待动作 0 和缺失动作 None 均返回 False。
    # 只检查目标格当前是否空闲，不预测别人的下一步，也不处理同时争抢空格。
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
    # 输入：observation 提供 obstacles 局部障碍矩阵；rnd 提供随机抽样和打乱功能。
    # 输出：动作编号 0～4，其中 0 为等待；随机移动方向只保证不指向静态障碍物。
    # 无候选时：50% 等待，50% 尝试随机方向；四周全是障碍物时仍等待。
    if rnd.random() <= 0.5:
        return 0

    actions = [1, 2, 3, 4]
    rnd.shuffle(actions)
    obstacles = np.asarray(observation["obstacles"])
    i, j = obstacles.shape[0] // 2, obstacles.shape[1] // 2
    for action in actions:
        di, dj = _MOVES[action]
        # 保留原后备规则：只排除静态障碍物，不排除当前可见的其他智能体。
        if obstacles[i + di, j + dj] == 0:
            return action
    return 0


# 第二部分：在累计看到的静态地图上搜索，忽略其他智能体和动态失败缓存。
class StaticAStarCheck:
    def __init__(self, max_steps: int = INF):
        # 输入：max_steps 为单次静态 A* 搜索最多扩展的节点数。
        # 输出：无返回值；保存搜索上限，静态规划器在首次 observe 时创建。
        self.max_steps = int(max_steps)
        self._planners = None

    def observe(self, observations):
        # 输入：按智能体排列的观测列表，使用每项的 obstacles 和 xy。
        # 输出：无返回值；为每个智能体积累其看到的静态障碍物，不记录其他智能体位置。
        count = len(observations)
        if self._planners is None:
            self._planners = [planner(self.max_steps) for _ in range(count)]
        elif len(self._planners) != count:
            raise ValueError("Static A* agent count changed without reset.")
        for local_planner, observation in zip(self._planners, observations):
            # 每步积累地图，不是等到检测到反向才开始记录。
            _update_map(local_planner, observation, [])

    def get_action(self, index, observation):
        # 输入：index 为智能体下标；observation 提供该智能体的位置 xy 和目标 target_xy。
        # 输出：静态 A* 路径首步的动作编号；无完整路径时返回 None，此处尚不检查当前占用。
        if self._planners is None:
            raise RuntimeError("Observe the agent batch before querying static A*.")
        local_planner = self._planners[index]
        position = tuple(map(int, observation["xy"]))
        target = tuple(map(int, observation["target_xy"]))
        local_planner.update_static_path(position, target)
        # False：静态查询不使用 BestMove；找不到完整路径就返回 None。
        action = _path_action(local_planner.get_next_node(False))
        # 静态查询只提供建议，不单独记录动作执行的成功或失败。
        local_planner.cancel_desired()
        return action


# 第三部分：检查候选是否反向，必要时用静态 A* 的建议替换。
class AORePlanWrapper:
    def __init__(self, agent, max_steps: int = INF):
        # 输入：agent 为提供动态候选的 AORePlanBase；max_steps 为静态搜索的节点扩展上限。
        # 输出：无返回值；初始化静态查询器、上一时刻位置、上次目标和候选替换记录。
        self.agent = agent
        self.rnd = agent.rnd
        self.static_astar = StaticAStarCheck(max_steps=max_steps)
        self.previous_position = None
        self.last_target = None
        self.last_dynamic_override_mask = None

    def _ensure_state(self, count):
        # 输入：count 为当前观测中的智能体数量。
        # 输出：无返回值；首次调用时创建位置和目标历史，后续调用检查人数是否一致。
        if self.previous_position is None:
            self.previous_position = [None] * count
            self.last_target = [None] * count
            return
        if len(self.previous_position) != count:
            raise ValueError("AORePlan agent count changed without reset.")

    def _static_astar_action(self, index, observation):
        # 输入：index 为智能体下标；observation 提供位置、目标、障碍物和其他智能体观测。
        # 输出：通过当前占用检查的静态动作；无路径或首步不可用时返回等待动作 0。
        action = self.static_astar.get_action(index, observation)
        # 静态建议无路径，或首步格子当前被占用，就主动改为等待再交给模拟器。
        return int(action) if _local_cell_is_free(observation, action) else 0

    @staticmethod
    def _returns_to_previous_position(position, action, previous):
        # 输入：position 为当前位置，action 为候选动作，previous 为上一时刻位置或 None。
        # 输出：动作会移回上一时刻位置时返回 True；否则返回 False。
        # 反向：移动后回到上一时刻的位置，而不是上一个“不同”的位置。
        if action in (None, 0) or previous is None:
            return False
        dx, dy = _MOVES[int(action)]
        return (position[0] + dx, position[1] + dy) == previous

    def commit_proposals(self, executed_mask):
        # 输入：executed_mask 为同长度布尔列表，表示最终 AORePlan 动作是否被采用并提交。
        # 输出：无返回值；结合内部替换记录，通知底层哪些原始动态候选确实被提交。
        # 只有真正提交且未被替换的动态候选，才保留其执行反馈记录。
        self.agent.commit_proposals(
            [
                bool(executed) and not overridden
                for executed, overridden in zip(executed_mask, self.last_dynamic_override_mask)
            ]
        )

    def act(self, observations, skip_agents=None):
        # 输入：按智能体排列的观测列表（obstacles、agents、xy、target_xy），以及可选跳过列表。
        # 输出：同长度的最终动作列表，0 为等待、1～4 为移动；被跳过的智能体对应 None。
        #       同时更新位置与目标历史、静态地图，以及本步哪些动态候选被替换的记录。
        # 先产生动态候选，再逐个检查；BestMove 的候选也会经过下面的反向检测。
        actions = list(self.agent.act(observations, skip_agents=skip_agents))
        self._ensure_state(len(actions))

        self.static_astar.observe(observations)
        self.last_dynamic_override_mask = [False] * len(actions)

        for index, observation in enumerate(observations):
            raw_action = actions[index]
            position = tuple(map(int, observation["xy"]))
            target = tuple(map(int, observation["target_xy"]))
            if self.last_target[index] != target:
                # 新目标不沿用旧目标的反向判断历史，但保留累计静态地图。
                self.previous_position[index] = None
                self.last_target[index] = target

            previous = self.previous_position[index]
            # 每步记录实际位置，包括等待或被阻挡后仍在原地的步。
            self.previous_position[index] = position

            if skip_agents is not None and bool(skip_agents[index]):
                continue

            if position == target:
                actions[index] = 0
                continue

            if raw_action is None:
                # 随机后备动作直接返回，不再做反向检测或静态 A* 查询。
                actions[index] = original_random_or_stay(observation, self.rnd)
                self.last_dynamic_override_mask[index] = True
                continue

            if not self._returns_to_previous_position(position, raw_action, previous):
                continue

            static_action = self._static_astar_action(index, observation)
            # 静态建议不反向才替换（等待也可替换）；仍反向时保留原动态候选。
            # 因此 AORePlan 减少无效反向，但不会禁止所有反向移动。
            if not self._returns_to_previous_position(position, static_action, previous):
                actions[index] = int(static_action)
                self.last_dynamic_override_mask[index] = True

        return actions
