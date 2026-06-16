"""Stanley steering controller.

This module contains the reusable Stanley controller logic from the thesis
workflow. It computes steering commands from local waypoint geometry using
heading error, cross-track error, yaw-rate damping, and steering smoothing terms.
"""

from math import atan2
from typing import Any

import numpy as np
import pandas as pd


def quaternion_to_yaw(q: Any) -> float:
    """Convert a quaternion-like object to yaw/heading in radians.

    The input is expected to have x, y, z, and w attributes.
    """
    return float(
        np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2),
        )
    )


def parse_path_df(df_path: pd.DataFrame) -> pd.DataFrame:
    """Parse a ROS/Foxglove path DataFrame where each Path message has one pose.

    Adds position, quaternion, and heading columns.
    """
    rows = []

    for _, row in df_path.iterrows():
        pose_stamped = row["poses"][0]
        position = pose_stamped.pose.position
        orientation = pose_stamped.pose.orientation

        rows.append(
            {
                "timestamp_ns": row["timestamp_ns"],
                "frame_id": row.get("frame_id", None),
                "x": position.x,
                "y": position.y,
                "z": position.z,
                "qx": orientation.x,
                "qy": orientation.y,
                "qz": orientation.z,
                "qw": orientation.w,
                "heading": quaternion_to_yaw(orientation),
            }
        )

    return pd.DataFrame(rows)


class StanleyController:
    """Stanley-style steering controller."""

    def __init__(
        self,
        yaw_error_gain: float = 0.2,
        control_gain: float = 2.5,
        k_soft: float = 1.0,
        k_yaw_rate: float = 0.0,
        k_damp_steer: float = 0.0,
        max_steer: float = np.deg2rad(24),
        wheelbase: float = 0.36,
        yaw_angle: float = np.pi / 2,
    ) -> None:
        self.k_yaw_error = yaw_error_gain
        self.k = control_gain
        self.k_soft = k_soft
        self.k_yaw_rate = k_yaw_rate
        self.k_damp_steer = k_damp_steer
        self.max_steer = max_steer
        self.wheelbase = wheelbase
        self.yaw_angle = yaw_angle

    def calculate_yaw_term(self, path_angle: float) -> float:
        """Calculate the yaw error term for the Stanley controller."""
        path_angle = -path_angle  # Correct heading for y-axis flip.
        yaw_error = self.yaw_angle - path_angle
        return float(yaw_error)

    def calculate_crosstrack_term(
        self,
        target_velocity: float,
        x: float,
        y: float,
        yaw_angle: float,
    ) -> tuple[float, float]:
        """Calculate cross-track steering correction and signed cross-track error."""
        # Shift from vehicle center to front axle.
        front_x = self.wheelbase
        front_y = 0.0

        error_x = x - front_x
        error_y = y - front_y

        # Path normal from yaw angle.
        normal_x = -np.sin(yaw_angle)
        normal_y = np.cos(yaw_angle)

        crosstrack_error = error_x * normal_x + error_y * normal_y

        crosstrack_steering_error = atan2(
            self.k * crosstrack_error,
            self.k_soft + target_velocity,
        )

        return float(crosstrack_steering_error), float(crosstrack_error)

    def calculate_yaw_rate_term(
        self,
        target_velocity: float,
        steering_angle: float,
    ) -> tuple[float, float]:
        """Calculate yaw-rate damping term from a bicycle-model yaw-rate estimate."""
        yaw_rate = target_velocity * np.sin(steering_angle) / self.wheelbase
        yaw_rate_error = self.k_yaw_rate * (-yaw_rate)

        return float(yaw_rate_error), float(yaw_rate)

    def calculate_steering_delay_term(
        self,
        computed_steering_angle: float,
        previous_steering_angle: float,
    ) -> float:
        """Smooth steering changes using a weighted previous-steering term."""
        steering_delay_error = self.k_damp_steer * (
            previous_steering_angle - computed_steering_angle
        )

        return float(steering_delay_error)

    def stanley_control(
        self,
        x: float,
        y: float,
        yaw_angle: float,
        target_velocity: float,
        current_steering_angle: float,
    ) -> float:
        """Compute a steering command in radians."""
        yaw_error = self.calculate_yaw_term(yaw_angle)

        crosstrack_steering_error, _ = self.calculate_crosstrack_term(
            target_velocity,
            x,
            y,
            yaw_angle,
        )

        yaw_rate_damping, _ = self.calculate_yaw_rate_term(
            target_velocity,
            current_steering_angle,
        )

        steering_angle = (
            self.k_yaw_error * yaw_error
            + crosstrack_steering_error
            + yaw_rate_damping
        )

        steering_angle += self.calculate_steering_delay_term(
            steering_angle,
            current_steering_angle,
        )

        steering_angle = np.clip(
            steering_angle,
            -self.max_steer,
            self.max_steer,
        )

        return float(steering_angle)
