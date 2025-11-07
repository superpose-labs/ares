"""
Sub-trajectory windowing for behavior clustering.

This module extracts fixed-size sliding windows from trajectories to capture
fine-grained behaviors instead of whole-trajectory patterns.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Configuration for window extraction."""
    window_size: int = 10  # Number of timesteps per window
    stride: int = 5  # Stride between windows (50% overlap)
    source: str = "state"  # "state", "action", or "both"
    flatten: bool = True  # Flatten window to 1D vector


class TrajectoryWindower:
    """
    Extract sliding windows from trajectories.

    For a trajectory of length T with window size W and stride S:
    - Number of windows: (T - W) / S + 1
    - Window i starts at position i * S and spans [i*S : i*S + W]

    Example:
        T=100, W=10, S=5 → 19 windows
        Window 0: [0:10], Window 1: [5:15], ..., Window 18: [90:100]
    """

    def __init__(self, config: Optional[WindowConfig] = None):
        """
        Initialize windower with configuration.

        Args:
            config: Window extraction configuration (default: W=10, S=5, source="state")
        """
        self.config = config or WindowConfig()

    def extract_windows(
        self,
        state_trajectory: np.ndarray,
        action_trajectory: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract sliding windows from a single trajectory.

        Args:
            state_trajectory: State trajectory, shape (T, state_dim)
            action_trajectory: Action trajectory, shape (T, action_dim), optional

        Returns:
            windows: Window embeddings, shape (num_windows, window_dim)
            window_positions: Starting position of each window in the trajectory

        Raises:
            ValueError: If trajectory is too short for window size
        """
        T = state_trajectory.shape[0]
        W = self.config.window_size
        S = self.config.stride

        if T < W:
            raise ValueError(f"Trajectory length {T} < window size {W}")

        # Calculate number of windows
        num_windows = (T - W) // S + 1

        # Select source data
        if self.config.source == "state":
            data = state_trajectory
        elif self.config.source == "action":
            if action_trajectory is None:
                raise ValueError("Action trajectory required for source='action'")
            data = action_trajectory
        elif self.config.source == "both":
            if action_trajectory is None:
                raise ValueError("Action trajectory required for source='both'")
            data = np.concatenate([state_trajectory, action_trajectory], axis=1)
        else:
            raise ValueError(f"Invalid source: {self.config.source}")

        # Extract windows
        windows = []
        window_positions = []

        for i in range(num_windows):
            start = i * S
            end = start + W
            window = data[start:end]  # Shape: (W, feature_dim)

            if self.config.flatten:
                window = window.flatten()  # Shape: (W * feature_dim,)

            windows.append(window)
            window_positions.append(start)

        return np.array(windows), window_positions

    def extract_all_windows(
        self,
        trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """
        Extract windows from multiple trajectories.

        Args:
            trajectories: Dict mapping rollout_id → (state_traj, action_traj)

        Returns:
            all_windows: Stacked windows, shape (total_windows, window_dim)
            window_to_rollout: List of rollout IDs for each window
            window_positions: List of starting positions for each window

        Example:
            If rollout_0 has 19 windows and rollout_1 has 19 windows:
            - all_windows.shape = (38, window_dim)
            - window_to_rollout = ['rollout_0'] * 19 + ['rollout_1'] * 19
            - window_positions = [0,5,10,...,90] + [0,5,10,...,90]
        """
        all_windows = []
        window_to_rollout = []
        all_positions = []

        for rollout_id, (state_traj, action_traj) in trajectories.items():
            try:
                windows, positions = self.extract_windows(state_traj, action_traj)
                all_windows.append(windows)
                window_to_rollout.extend([rollout_id] * len(windows))
                all_positions.extend(positions)
            except ValueError as e:
                print(f"Warning: Skipping {rollout_id}: {e}")
                continue

        if not all_windows:
            raise ValueError("No valid windows extracted from trajectories")

        return np.vstack(all_windows), window_to_rollout, all_positions

    @property
    def num_windows_per_trajectory(self) -> int:
        """
        Calculate number of windows per trajectory (assuming T=100).

        For KAIST dataset: T=100 timesteps (interpolated)
        """
        T = 100  # Standard trajectory length after interpolation
        W = self.config.window_size
        S = self.config.stride
        return (T - W) // S + 1

    @property
    def window_dim(self) -> int:
        """
        Calculate window dimension.

        For KAIST dataset:
        - state_dim = 21
        - action_dim = 20
        - Window size W = 10

        Dimension depends on source:
        - "state": W * 21 = 210
        - "action": W * 20 = 200
        - "both": W * (21 + 20) = 410
        """
        # This is a helper method - actual dimension depends on data
        # Will be calculated during extraction
        raise NotImplementedError("Use actual window shape after extraction")


def compute_window_statistics(
    all_windows: np.ndarray,
    window_to_rollout: List[str],
    num_rollouts: int,
) -> Dict[str, any]:
    """
    Compute statistics about extracted windows.

    Args:
        all_windows: All windows, shape (total_windows, window_dim)
        window_to_rollout: Rollout ID for each window
        num_rollouts: Total number of rollouts

    Returns:
        Dictionary with statistics
    """
    total_windows = len(all_windows)
    windows_per_rollout = total_windows / num_rollouts
    window_dim = all_windows.shape[1]

    # Count windows per rollout
    from collections import Counter
    rollout_counts = Counter(window_to_rollout)

    stats = {
        'total_windows': total_windows,
        'num_rollouts': num_rollouts,
        'avg_windows_per_rollout': windows_per_rollout,
        'window_dim': window_dim,
        'min_windows_per_rollout': min(rollout_counts.values()),
        'max_windows_per_rollout': max(rollout_counts.values()),
        'unique_rollouts': len(rollout_counts),
    }

    return stats


if __name__ == "__main__":
    # Example usage
    print("=== Sub-Trajectory Windowing Example ===\n")

    # Simulate a trajectory
    T = 100
    state_dim = 21
    action_dim = 20

    state_traj = np.random.randn(T, state_dim)
    action_traj = np.random.randn(T, action_dim)

    # Create windower with default config (W=10, S=5, source="state")
    windower = TrajectoryWindower()

    print(f"Configuration:")
    print(f"  Window size: {windower.config.window_size}")
    print(f"  Stride: {windower.config.stride}")
    print(f"  Source: {windower.config.source}")
    print(f"  Flatten: {windower.config.flatten}\n")

    # Extract windows
    windows, positions = windower.extract_windows(state_traj, action_traj)

    print(f"Trajectory shape: ({T}, {state_dim})")
    print(f"Number of windows: {len(windows)}")
    print(f"Window shape: {windows.shape}")
    print(f"Window positions: {positions[:5]} ... {positions[-3:]}\n")

    # Test with multiple trajectories
    trajectories = {
        f'rollout_{i}': (
            np.random.randn(T, state_dim),
            np.random.randn(T, action_dim)
        )
        for i in range(5)
    }

    all_windows, window_to_rollout, all_positions = windower.extract_all_windows(trajectories)

    print(f"Multiple trajectories:")
    print(f"  Total windows: {len(all_windows)}")
    print(f"  Windows shape: {all_windows.shape}")
    print(f"  Sample window IDs: {window_to_rollout[:5]} ... {window_to_rollout[-3:]}\n")

    # Compute statistics
    stats = compute_window_statistics(all_windows, window_to_rollout, len(trajectories))
    print("Window statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
