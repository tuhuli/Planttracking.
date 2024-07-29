from math import sin, cos, radians

import cv2
import numpy as np


def ellipse_formula(x, y, a, b, h, k, angle):
    return (((((x - h) * cos(angle) + (y - k) * sin(angle)) ** 2) / (a ** 2)) +
            ((((x - h) * sin(angle) - (y - k) * cos(angle)) ** 2) / (b ** 2)))


def create_synthetic_video(number_of_frames, video_location):
    width, height = 672, 368
    fps = 59.94
    ellipse_axes = (25, 80)
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
