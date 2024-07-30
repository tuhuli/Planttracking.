from math import sin, cos, radians

import cv2
import numpy as np


def ellipse_formula(x: float, y: float, a: float, b: float, h: float, k: float, angle: float) -> float:
    """
        Calculate the value of the ellipse formula for given coordinates and ellipse parameters.
        Values bigger then 1 does not lie in the ellipsis

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


def create_synthetic_video(number_of_frames: int, video_location: str) -> None:
    """
        Create a synthetic video with moving ellipses.

        Parameters:
        number_of_frames (int): The total number of frames in the video.
        video_location (str): The file path where the video will be saved.
        """
    width, height = 672, 368
    fps = 59.94
    ellipse_axes = (50, 110)
    frames_between_objects = 100

    start_x_pos = 0
    start_angle = radians(-45)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_location, fourcc, fps, (width, height), isColor=False)

    objects = []
    for frame_num in range(number_of_frames):
        if frame_num % frames_between_objects == 0:
            objects.append((start_x_pos, start_angle))

        frame = np.zeros((height, width), dtype=np.uint8)
        new_objects = []
        for i in range(len(objects)):
            x_pos, angle = objects[i]

            x_pos += 5
            y_pos = int(height / 1.5)
            angle += 0.010

            if x_pos < 750:
                new_objects.append((x_pos, angle))

            for y in range(-ellipse_axes[1], ellipse_axes[1]):
                for x in range(-ellipse_axes[1], ellipse_axes[1]):
                    intensity = ellipse_formula(x, y, ellipse_axes[0], ellipse_axes[1], 1, 1, angle)

                    if intensity < 1 and 0 <= x_pos + x - ellipse_axes[0] < width and 0 <= y_pos + y - ellipse_axes[
                        1] < height:
                        frame[y_pos + y - ellipse_axes[1], x_pos + x - ellipse_axes[0]] = int(255 * (1 - intensity))

        out.write(frame)
        objects = new_objects

    out.release()


create_synthetic_video(1200, "synthetic_video.mp4")
