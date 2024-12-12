from math import sin, cos, radians
from typing import Dict, List

import json
import cv2
import numpy as np


def ellipse_formula(x: float, y: float, a: float, b: float, h: float, k: float, angle: float) -> float:
    """
        Calculate the value of the ellipse formula for given coordinates and ellipse parameters.

        Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        a (float): The semi-major axis of the ellipse.
        b (float): The semi-minor axis of the ellipse.
        h (float): The x-coordinate of the ellipse center.
        k (float): The y-coordinate of the ellipse center.
        angle (float): The rotation angle of the ellipse in radians.

        Returns:
        float: The computed value of the ellipse formula at the point (x, y).
        """
    return (((((x - h) * cos(angle) + (y - k) * sin(angle)) ** 2) / (a ** 2)) +
            ((((x - h) * sin(angle) - (y - k) * cos(angle)) ** 2) / (b ** 2)))


def write_ellipse_to_frame(x_pos: int, y_pos: int, angle: float, frame: np.ndarray) -> None:
    """
    Draws an ellipse on a given frame.

    Parameters:
        x_pos (int): X-coordinate of the ellipse center.
        y_pos (int): Y-coordinate of the ellipse center.
        angle (float): Rotation angle of the ellipse in radians.
        frame (np.ndarray): The frame on which the ellipse is drawn.
    """
    height, width = frame.shape

    ellipse_axes = (50, 110)

    for y in range(y_pos - ellipse_axes[1], y_pos + ellipse_axes[1]):
        for x in range(x_pos - ellipse_axes[1], x_pos + ellipse_axes[1]):
            intensity = ellipse_formula(x, y, ellipse_axes[0], ellipse_axes[1], x_pos, y_pos, angle)

            if intensity < 1 and 0 <= x < width and 0 <= y < height:
                frame[y, x] = int(255 * (1 - intensity))


def create_ground_truth_file(ellipses_id_frames: Dict[str, List[Dict[str, float]]], location_path: str) -> None:
    """
    Create a ground truth JSON file containing the positions of ellipses.

    Parameters:
    ellipses_id_frames (Dict[str, List[Dict[str, float]]]): Dictionary containing the ground truth data
    location_path (str): The base file path (without extension) for the JSON file.

    """
    with open(f"{location_path}_ground_truth.json", "w") as file:
        json.dump(ellipses_id_frames, file, indent=4)


def create_synthetic_video(number_of_frames: int, video_location: str, velocity:int) -> None:
    """
        Create a synthetic video with moving ellipses.

        Parameters:
        number_of_frames (int): The total number of frames in the video.
        video_location (str): The file path where the video will be saved.
        """

    width, height = 672, 368
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_location, fourcc, 59.94, (width, height), isColor=False)

    start_x_pos = -50
    y_pos = 150
    start_angle = radians(-45)
    velocity_sign = 1

    id_counter = 1
    frames_with_ellipses = {}
    ellipses = []
    for frame_num in range(number_of_frames):
        print(frame_num)
        frame = np.zeros((height, width), dtype=np.uint8)
        new_ellipses = []

        if len(ellipses) == 0 or (ellipses[-1][0] >= 500 and len(ellipses) == 1):
            ellipses.append((id_counter, start_x_pos, start_angle))
            id_counter += 1

        for id, x_pos, angle in ellipses:
            x_pos += velocity
            angle = radians(-45) + (x_pos / width) * (radians(45) - radians(-45))
            if x_pos < 750:
                new_ellipses.append((id, x_pos, angle))

            write_ellipse_to_frame(x_pos, y_pos, angle, frame)

            if velocity >= 20:
                velocity_sign = -1
            if velocity <= 2:
                velocity_sign = 1
            #velocity += velocity_sign

            frames_with_ellipses[str(frame_num)].append({"id": id, "x": x_pos, "y": y_pos})

        out.write(frame)
        ellipses = new_ellipses

    out.release()

    create_ground_truth_file(frames_with_ellipses, video_location[:-4])


create_synthetic_video(1600, "20_velocity_synthetic.mp4", 20)
