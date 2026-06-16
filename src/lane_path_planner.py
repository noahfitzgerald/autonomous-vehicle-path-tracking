"""Lane-based local path planner.

This module contains the reusable path-planning logic from the thesis workflow.
It converts lane mask detections into local ground-plane waypoints for steering
controller evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ast
import cv2
import numpy as np
import yaml


@dataclass
class LaneData:
    """Computed information for a single detected lane line."""

    label: str
    success: bool = True

    # Set by get_pixel_coordinates()
    pixel_coordinates: Optional[tuple[np.ndarray, np.ndarray]] = None

    # Set by to_surface_coordinates()
    surface_coordinates: Optional[tuple[np.ndarray, np.ndarray]] = None

    # Set by get_base_segment()
    segment_coordinates: Optional[tuple[np.ndarray, np.ndarray]] = None

    # Set by estimate_road_direction()
    area: Optional[float] = None
    slope: Optional[float] = None
    heading: Optional[np.ndarray] = None
    theta: Optional[float] = None
    base: Optional[tuple[float, float]] = None
    projected_coordinates: Optional[tuple[np.ndarray, np.ndarray]] = None

    # Set by get_waypoints()
    waypoints: Optional[list[tuple[float, float, float]]] = None


class LanePathPlanner:
    """Generate local path waypoints from segmented lane mask data."""

    def __init__(
        self,
        config_path: str | Path,
        id2label: dict[int, str],
        half_road_width: float = 0.4,
        left_id: int = 4,
        right_id: int = 5,
        min_coordinates: int = 10,
    ) -> None:
        self.id2label = id2label
        self.half_road_width = half_road_width
        self.LEFT_ID = left_id
        self.RIGHT_ID = right_id
        self.min_coordinates = min_coordinates

        self.read_transform_config(config_path)
        self.reset_data()

    def reset_data(self) -> None:
        """Initialize or reset lane data from the ID-to-label mapping."""
        self.lanes: dict[int, LaneData] = {
            lane_id: LaneData(label=label)
            for lane_id, label in self.id2label.items()
        }

    def read_transform_config(self, filepath: str | Path) -> None:
        """Read homography and image metadata from a YAML config file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        homography = data.get("homography")

        if isinstance(homography, str):
            homography = ast.literal_eval(homography)

        if homography is None:
            raise ValueError("Config file must contain a 'homography' entry.")

        self.H = np.array(homography, dtype=float)
        self.H_inv = np.linalg.inv(self.H)
        self.im_w = data.get("image_width")
        self.im_h = data.get("image_height")

    def run_detection(
        self,
        mask: np.ndarray,
    ) -> tuple[bool, Optional[list[tuple[float, float, float]]], Optional[float]]:
        """Run lane detection and waypoint generation on one segmentation mask."""
        self.reset_data()

        mask = self.preprocess_mask(mask)

        self.get_pixel_coordinates(mask)
        self.to_surface_coordinates()
        self.get_base_segment(length=0.15, start=0.05)
        self.estimate_road_direction()
        success, waypoints, area = self.get_waypoints()

        return success, waypoints, area

    @staticmethod
    def preprocess_mask(mask: np.ndarray) -> np.ndarray:
        """Preprocess a lane mask before extracting lane coordinates."""
        return mask

    def is_successful(self) -> bool:
        """Return True if at least one lane line is currently valid."""
        left_success = self.lanes.get(self.LEFT_ID, LaneData("left", False)).success
        right_success = self.lanes.get(self.RIGHT_ID, LaneData("right", False)).success
        return left_success or right_success

    def get_pixel_coordinates(self, mask: np.ndarray) -> None:
        """Extract pixel coordinates for each lane ID from the mask."""
        for lane_id, lane in self.lanes.items():
            if not lane.success:
                continue

            v, u = np.where(mask == lane_id)
            lane.pixel_coordinates = (u, v)

            if len(u) < self.min_coordinates:
                lane.success = False

    def to_surface_coordinates(self) -> None:
        """Project lane pixel coordinates into ground-plane surface coordinates."""
        for lane in self.lanes.values():
            if not lane.success:
                continue

            if lane.pixel_coordinates is None:
                lane.success = False
                continue

            u, v = lane.pixel_coordinates

            if len(v) == 0:
                lane.success = False
                continue

            pixels = np.array([u, v, np.ones_like(u)])
            points = self.H @ pixels
            x, y = points[:2] / points[-1]

            sorted_indices = np.argsort(x)
            x = x[sorted_indices]
            y = y[sorted_indices]

            lane.surface_coordinates = (x, y)

            if len(x) < self.min_coordinates:
                lane.success = False

    def get_base_segment(self, length: float = 0.2, start: float = 0.1) -> None:
        """Extract a near-field lane segment used to estimate local direction."""
        for lane in self.lanes.values():
            if not lane.success:
                continue

            if lane.surface_coordinates is None:
                lane.success = False
                continue

            x, y = lane.surface_coordinates
            dx = x - x[0]
            dy = y - y[0]
            distances = np.sqrt(dx**2 + dy**2)

            total = start + length

            if np.any(distances > total):
                end_index = np.argmax(distances > total)
            else:
                end_index = len(distances)

            if np.any(distances > start):
                start_index = np.argmax(distances > start)
            else:
                start_index = 0

            xs = x[start_index:end_index]
            ys = y[start_index:end_index]

            lane.segment_coordinates = (xs, ys)

            if len(xs) < self.min_coordinates:
                lane.success = False

    @staticmethod
    def _get_heading(slope: float) -> np.ndarray:
        """Return a unit heading vector from a lane-line slope."""
        dx = 1.0
        dy = slope
        return np.array([dx, dy]) / np.linalg.norm([dx, dy])

    def estimate_road_direction(self) -> None:
        """Estimate local road heading from each valid lane line."""
        for lane in self.lanes.values():
            if not lane.success:
                continue

            if lane.surface_coordinates is None or lane.segment_coordinates is None:
                lane.success = False
                continue

            x, y = lane.surface_coordinates
            xs, ys = lane.segment_coordinates

            coeffs = np.polyfit(xs, ys, deg=1)
            slope = float(coeffs[0])

            x0 = float(xs[0])
            y0 = float(ys[0])

            heading = self._get_heading(slope)
            xp, yp, area = self._project(x, y, x0, y0, heading)

            lane.area = area
            lane.slope = slope
            lane.heading = heading
            lane.theta = float(np.arccos(np.dot(heading, [1, 0])))
            lane.base = (x0, y0)
            lane.projected_coordinates = (xp, yp)

    @staticmethod
    def _project(
        x: np.ndarray,
        y: np.ndarray,
        x0: float,
        y0: float,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Project lane points onto a line defined by a base point and direction."""
        d = direction
        v = np.column_stack((x - x0, y - y0))

        proj_lengths = np.dot(v, d) / np.dot(d, d)
        proj_vectors = np.outer(proj_lengths, d)

        projected_points = np.column_stack(
            (x0 + proj_vectors[:, 0], y0 + proj_vectors[:, 1])
        )

        xp, yp = projected_points.T
        area = float(np.trapz(y, x) - np.trapz(yp, xp))

        return xp, yp, area

    def get_waypoints(
        self,
    ) -> tuple[bool, Optional[list[tuple[float, float, float]]], Optional[float]]:
        """Generate a local centerline waypoint from the active lane."""
        left = self.lanes[self.LEFT_ID]
        right = self.lanes[self.RIGHT_ID]

        if not left.success and not right.success:
            return False, None, None

        if left.success and not right.success:
            active_lane = left
        elif not left.success and right.success:
            active_lane = right
        else:
            # Both detected: select the lane closest to the vehicle.
            x_l, _ = left.surface_coordinates
            x_r, _ = right.surface_coordinates
            active_lane = left if min(x_l) < min(x_r) else right

        if active_lane.heading is None or active_lane.base is None:
            return False, None, None

        r_ccw = np.array([[0, -1], [1, 0]])
        r_cw = np.array([[0, 1], [-1, 0]])

        is_right = active_lane is right

        if is_right:
            normal_vector = r_cw @ active_lane.heading
        else:
            normal_vector = r_ccw @ active_lane.heading

        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        x0, y0 = active_lane.base
        base = np.array([x0, y0])
        center = base + self.half_road_width * normal_vector

        cx = float(center[0])
        cy = float(center[1])
        theta = float(active_lane.theta)

        active_lane.waypoints = [(cx, cy, theta)]

        return True, active_lane.waypoints, active_lane.area

    def plot_data(
        self,
        axs,
        width_y: float = 1.5,
        distance_x: float = 2.5,
        image: Optional[np.ndarray] = None,
    ) -> None:
        """Plot extracted lane geometry and generated waypoint."""
        ax1, ax2 = axs

        light = {"line_left": "lightcoral", "line_right": "lightskyblue"}
        dark = {"line_left": "red", "line_right": "dodgerblue"}

        for lane in self.lanes.values():
            if not lane.success:
                continue

            label = lane.label

            if lane.surface_coordinates is not None:
                x, y = lane.surface_coordinates
                ax2.scatter(
                    x,
                    y,
                    s=0.1,
                    label=label,
                    zorder=1,
                    c=light.get(label, "lightgray"),
                )

            if lane.segment_coordinates is not None:
                xs, ys = lane.segment_coordinates
                ax2.plot(xs, ys, zorder=2, c=dark.get(label, "gray"))

            if lane.projected_coordinates is not None and lane.base is not None:
                x0, y0 = lane.base
                self._plot_vector(
                    ax2,
                    x0,
                    y0,
                    lane.theta,
                    length=0.4,
                    color="k",
                    zorder=3,
                )

            if lane.waypoints is not None:
                for cx, cy, theta in lane.waypoints:
                    self._plot_vector(
                        ax2,
                        cx,
                        cy,
                        theta,
                        length=0.4,
                        color="k",
                        label="center",
                        zorder=3,
                    )

                    if image is not None:
                        pts = self._compute_line(cx, cy, theta)
                        pix = self._project_points_on_image(pts)
                        image = self._draw_polyline_on_image(
                            image,
                            pix,
                            color=(0, 255, 0),
                        )

        if image is not None:
            ax1.imshow(image)

        ax2.axis("equal")
        ax2.grid()
        ax2.set_ylim(-width_y, width_y)
        ax2.set_xlim(0, distance_x)
        ax2.legend()

    @staticmethod
    def _compute_line(
        x: float,
        y: float,
        theta: float,
        length: float = 0.2,
    ) -> np.ndarray:
        dx, dy = np.array([np.cos(theta), np.sin(theta)]) * length
        return np.array([[x, y], [x + dx, y + dy]])

    def _project_points_on_image(self, points: np.ndarray) -> np.ndarray:
        points = np.array(points).reshape(-1, 2)
        homogeneous_points = np.column_stack((points, np.ones(len(points))))

        projected_points = homogeneous_points @ self.H_inv.T
        projected_points /= projected_points[:, 2].reshape(-1, 1)

        return projected_points[:, :2].astype(int)

    @staticmethod
    def _draw_polyline_on_image(
        image: np.ndarray,
        points: np.ndarray,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=False, color=color, thickness=thickness)
        return image

    @staticmethod
    def _plot_vector(
        ax,
        x0: float,
        y0: float,
        theta: float,
        length: float = 0.2,
        color: str = "r",
        label: Optional[str] = None,
        zorder: int = -1,
    ) -> None:
        dx, dy = np.array([np.cos(theta), np.sin(theta)]) * length

        ax.arrow(
            x0,
            y0,
            dx,
            dy,
            head_width=0.1 * length,
            head_length=0.1 * length,
            fc=color,
            ec=color,
            length_includes_head=True,
            label=label,
            zorder=zorder,
        )
