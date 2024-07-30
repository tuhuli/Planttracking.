from typing import List

import cv2
import numpy as np

from object_detection import get_plants_and_initialize_filter
from particle_filter import ParticleFilter
from visualization import bb_from_particle_filter, show_particles_in_image

"""
Same function is currently in Kalman test -> rewrite it to somewhere 
"""


def remove_duplicate_filters(filters):
    for p_filter in filters:
        for p_filter2 in filters:
            if p_filter == p_filter2:
                continue
            if abs(p_filter.x - p_filter2.x) < 50:
                filters.remove(p_filter2)


def remove_filter_outside_of_image(filters):
    for p_f in filters:
        if p_f.x > p_f.max_width or p_f.y > p_f.max_height or p_f.x < 0 or p_f.y < 0:
            filters.remove(p_f)


def particle_detection_on_video(new_cap: cv2.VideoCapture, read_cap: cv2.VideoCapture, out: cv2.VideoWriter,
                                grayscale: bool, show_particles: bool) -> None:
    """
    Applies Particle filter-based object detection on a video and saves the output.

    Parameters:
        new_cap (cv2.VideoCapture): Capture of output video.
        read_cap (cv2.VideoCapture): Capture of input video.
        out (cv2.VideoWriter): writer to save the output video with bounding boxes.
        grayscale(bool): boolean to signal grayscale video
        show_particles(bool): boolean to signal if the particles should be shown in the video
    """
    no_object_frames_counter = 10
    num_of_objects = 0
    filters: List[ParticleFilter] = []

    while True:
        out_ret, out_frame = new_cap.read()
        in_ret, in_frame = read_cap.read()

        if not out_ret or not in_ret:
            break

        threshold = 200
        thresh_image = np.where(in_frame < threshold, 0, in_frame)

        # Turn image to grayscale
        g_thresh_image = np.array(thresh_image[:, :, 0])
        g_image = np.array(in_frame[:, :, 0])
        plants, no_object_frames_counter, num_of_objects = get_plants_and_initialize_filter(g_thresh_image,
                                                                                            filters,
                                                                                            num_of_objects,
                                                                                            no_object_frames_counter,
                                                                                            'particle')
        for p_f in filters:
            p_f.predict((0.2, 0.05))
            p_f.convert_particles_to_int()
            p_f.update_with_image(g_image)
            # p_f.print_best_weights(300)

            mu, var = p_f.estimate()

            if p_f.neff() < p_f.number_of_particles / 2:
                p_f.basic_resample_weights()
                assert np.allclose(p_f.weights, 1 / p_f.number_of_particles)
            print(f"position: {mu}")

        remove_duplicate_filters(filters)
        remove_filter_outside_of_image(filters)

        if grayscale:
            image = in_frame
        else:
            image = out_frame

        image = bb_from_particle_filter(image, filters, grayscale)
        if show_particles:
            image = show_particles_in_image(image, filters, grayscale)

        # io.imshow(image)
        # io.show()
        out.write(image)
