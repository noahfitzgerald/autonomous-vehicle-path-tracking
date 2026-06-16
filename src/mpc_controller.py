"""Model Predictive Controller for local steering control.

This module contains the reusable MPC logic from the thesis workflow. It uses a
kinematic bicycle model and constrained optimization to compute steering commands
for local waypoint tracking.
"""

import numpy as np
from scipy.optimize import minimize


def make_local_waypoints(
    x_goal: float,
    y_goal: float,
    theta_goal: float,
    n: int = 5,
) -> np.ndarray:
    """Generate a simple local reference trajectory to a goal waypoint."""
    xs = np.linspace(0.0, x_goal, n)
    ys = np.linspace(0.0, y_goal, n)
    headings = np.linspace(0.0, theta_goal, n)

    return np.column_stack([xs, ys, headings])


class MPCController:
    """Simplified steering MPC using a kinematic bicycle model."""

    def __init__(
        self,
        wheelbase: float = 0.36,
        dt: float = 0.1,
        velocity: float = 1.0,
        max_steer: float = np.deg2rad(24),
        q_position: float = 15.0,
        q_heading: float = 4.0,
        r_steering: float = 0.5,
        r_steering_change: float = 2.0,
    ) -> None:
        self.wheelbase = wheelbase
        self.dt = dt
        self.velocity = velocity
        self.max_steer = max_steer

        self.q_position = q_position
        self.q_heading = q_heading
        self.r_steering = r_steering
        self.r_steering_change = r_steering_change

    @staticmethod
    def angle_wrap(angle: float) -> float:
        """Wrap an angle to the range [-pi, pi]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def predict_next_state(
        self,
        state: np.ndarray,
        steering_angle: float,
    ) -> np.ndarray:
        """Predict the next vehicle state using a kinematic bicycle model."""
        theta = state[2]

        next_state = np.zeros(3)

        next_state[0] = state[0] + self.velocity * np.cos(theta) * self.dt
        next_state[1] = state[1] + self.velocity * np.sin(theta) * self.dt
        next_state[2] = theta + (
            self.velocity / self.wheelbase
        ) * np.tan(steering_angle) * self.dt

        return next_state

    def calculate_cost(
        self,
        steering_sequence: np.ndarray,
        initial_state: np.ndarray,
        waypoints: np.ndarray,
    ) -> float:
        """Evaluate the MPC objective over the prediction horizon."""
        total_cost = 0.0
        state = initial_state.copy()
        previous_steering = 0.0

        for i, steering_angle in enumerate(steering_sequence):
            state = self.predict_next_state(state, steering_angle)

            position_error = np.sum((state[:2] - waypoints[i, :2]) ** 2)
            heading_error = self.angle_wrap(state[2] - waypoints[i, 2]) ** 2
            steering_effort = steering_angle**2
            steering_change = (steering_angle - previous_steering) ** 2

            total_cost += (
                self.q_position * position_error
                + self.q_heading * heading_error
                + self.r_steering * steering_effort
                + self.r_steering_change * steering_change
            )

            previous_steering = steering_angle

        return float(total_cost)

    def optimize_control(
        self,
        initial_state: np.ndarray,
        waypoints: np.ndarray,
    ):
        """Solve the constrained MPC steering optimization problem."""
        horizon = len(waypoints)

        initial_steering_sequence = np.zeros(horizon)

        bounds = [(-self.max_steer, self.max_steer)] * horizon

        result = minimize(
            self.calculate_cost,
            initial_steering_sequence,
            args=(initial_state, waypoints),
            bounds=bounds,
            method="SLSQP",
        )

        return result

    def mpc_control(
        self,
        initial_state: np.ndarray,
        waypoints: np.ndarray,
    ) -> float:
        """Compute the first optimal steering command from the MPC solution."""
        result = self.optimize_control(initial_state, waypoints)

        if result.success:
            steering_angle = result.x[0]
        else:
            steering_angle = 0.0

        steering_angle = np.clip(
            steering_angle,
            -self.max_steer,
            self.max_steer,
        )

        return float(steering_angle)
