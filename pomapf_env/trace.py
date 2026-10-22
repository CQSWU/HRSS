import numpy as np


class SharedTrace:
    def __init__(self, rho=0.1):
        self.rho = float(rho)
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1].")
        self.decay = 1.0 - self.rho

        self.tau = None
        self.prev_positions = None
        self._obstacle_mask = None

    def configure(self, height, width, clear=False):
        shape = (int(height), int(width))
        if self.tau is None or self.tau.shape != shape:
            self.tau = np.zeros(shape, dtype=np.float32)
        elif clear:
            self.tau.fill(0.0)
        return self.tau

    def configure_from_grid_config(self, grid_config, clear=False):
        return self.configure_from_obstacle_mask(
            self._grid_to_obstacles(grid_config.map), clear=clear
        )

    def configure_from_obstacle_mask(self, obstacle_mask, clear=False):
        obstacle_mask = np.asarray(obstacle_mask, dtype=bool)
        result = self.configure(*obstacle_mask.shape, clear=clear)

        if self._obstacle_mask is None or not np.array_equal(
            self._obstacle_mask, obstacle_mask
        ):
            self._obstacle_mask = obstacle_mask.copy()
            self.tau[self._obstacle_mask] = 0.0
        return result

    def clear(self):
        if self.tau is not None:
            self.tau.fill(0.0)

        self.prev_positions = None

    def reset_episode(
        self,
        obs_batch,
        positions=None,
        radius=None,
    ):
        self._ensure_runtime(obs_batch)
        self.tau.fill(0.0)

        positions = self._positions(positions)
        self._update_trace(positions, evaporate=False)
        self.prev_positions = positions
        self.add_tau_observation(
            obs_batch,
            positions=positions,
            radius=radius,
        )

    def observe_for_inference(
        self,
        obs_batch,
        positions=None,
        radius=None,
    ):
        self._ensure_runtime(obs_batch)
        positions = self._positions(positions)
        if self.prev_positions is None:
            self.reset_episode(
                obs_batch,
                positions=positions,
                radius=radius,
            )
            return

        self._update_trace(positions)
        self.prev_positions = positions
        self.add_tau_observation(
            obs_batch,
            positions=positions,
            radius=radius,
        )

    def extract_local_tau(self, x, y, radius):
        # Return a local trace crop centered by its traversable-cell mean.
        local, free_mask = self._local_tau_and_free_mask(x, y, int(radius))
        return self._relative_pressure(local, free_mask)

    def add_tau_observation(
        self,
        obs_batch,
        positions=None,
        radius=None,
    ):
        self._ensure_runtime(obs_batch)
        positions = self._positions(positions)

        for obs, (x, y) in zip(obs_batch, positions):
            x, y = int(x), int(y)
            local_radius = (
                self._observation_radius(obs) if radius is None else int(radius)
            )
            tau_local = self.extract_local_tau(x, y, local_radius)
            obs["tau"] = tau_local[np.newaxis, ...].astype(
                np.float32,
                copy=False,
            )

    @staticmethod
    def _observation_radius(observation):
        if "obs" in observation:
            return int(observation["obs"].shape[-1] // 2)
        if "obstacles" in observation:
            return int(observation["obstacles"].shape[-1] // 2)
        raise KeyError("Tau observations require either 'obs' or 'obstacles'.")

    def _local_tau_and_free_mask(self, x, y, radius):
        size = 2 * radius + 1
        free_mask = np.zeros((size, size), dtype=bool)
        local = np.zeros((size, size), dtype=np.float32)
        height, width = self.tau.shape
        x, y = int(x), int(y)
        x0, x1 = max(0, x - radius), min(height, x + radius + 1)
        y0, y1 = max(0, y - radius), min(width, y + radius + 1)

        if x0 < x1 and y0 < y1:
            lx0 = x0 - (x - radius)
            ly0 = y0 - (y - radius)
            crop = np.s_[lx0 : lx0 + (x1 - x0), ly0 : ly0 + (y1 - y0)]
            local[crop] = self.tau[x0:x1, y0:y1]
            free_mask[crop] = (
                True
                if self._obstacle_mask is None
                else ~self._obstacle_mask[x0:x1, y0:y1]
            )

        return local, free_mask

    def _ensure_runtime(self, obs_batch):
        if self.tau is None:
            raise RuntimeError("ACO state is not configured yet.")
        if self.prev_positions is not None and len(self.prev_positions) != len(
            obs_batch
        ):
            self.prev_positions = None

    @staticmethod
    def _positions(positions):
        if positions is None:
            raise RuntimeError(
                "Shared trace memory requires global agent positions. Raw "
                "observation coordinates are relative to each agent's initial "
                "position and cannot index the global trace map."
            )
        return np.rint(np.asarray(positions, dtype=np.float32)).astype(np.int64)

    def _update_trace(self, positions, evaporate=True):
        # Decay old traces and add one to each currently occupied cell.
        if evaporate:
            self.tau *= self.decay
        if self._obstacle_mask is not None:
            self.tau[self._obstacle_mask] = 0.0

        height, width = self.tau.shape
        occupied = {(int(x), int(y)) for x, y in positions}
        for x, y in occupied:
            if not (0 <= x < height and 0 <= y < width) or self._is_obstacle(x, y):
                continue

            self.tau[x, y] += 1.0

    @staticmethod
    def _relative_pressure(local, free_mask):
        # Exclude obstacles and padding from the mean and leave their output at zero.
        free_values = local[free_mask]
        if free_values.size == 0:
            return np.zeros_like(local, dtype=np.float32)

        mean = float(free_values.mean())
        pressure = local - mean
        pressure[~free_mask] = 0.0
        return pressure.astype(np.float32, copy=False)

    @staticmethod
    def _grid_to_obstacles(grid_map):
        if isinstance(grid_map, str):
            rows = [row for row in grid_map.splitlines() if row]
        else:
            rows = list(grid_map)

        height = len(rows)
        width = len(rows[0])
        obstacle_mask = np.zeros((height, width), dtype=bool)
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                obstacle_mask[i, j] = cell not in (0, ".", "0", False)
        return obstacle_mask

    def _is_obstacle(self, x, y):
        return self._obstacle_mask is not None and self._obstacle_mask[x, y]
