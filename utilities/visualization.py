from typing import List

import cv2
import numpy as np
from filters.kalman_filter.kalman_filter import KalmanFilterID
from filters.particle_filter.particle_filter import ParticleFilter
from utilities.evaluation import SyntheticEvaluator


def line_from_filter(image: np.ndarray, filters: List[ParticleFilter] | List[KalmanFilterID],
                     grayscale: bool) -> np.ndarray:
    """
    Draws vertical lines in the x position of filters and ID.

    Parameters:
        image (np.ndarray): The input image.
        filters (List[ParticleFilter] | List[KalmanFilterID]): List of filters.
        grayscale (bool): Boolean representing, if the image is grayscale.

    Returns:
        np.ndarray: Image with lines and IDs of filters.
    """

    for tracking_filter in filters:
        if grayscale:
            x = int(tracking_filter.get_centre_x())
        else:
            x = tracking_filter.get_centre_x() * 2

        image = cv2.line(image, (x, 0), (x, image.shape[0]), (255, 0, 255), 2)
        text_position = (int(x), int(100))
        cv2.putText(image, str(tracking_filter.id), text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    return image


def show_particles_in_image(frame: np.ndarray, particle_filters: List[ParticleFilter], grayscale: bool) -> np.ndarray:
    """
        Draws particles of filters.

        Args:
            frame (np.ndarray): The input image.
            particle_filters (List[ParticleFilter]): List of ParticleFilters.
            grayscale (bool): Boolean representing, if the image is grayscale.

        Returns:
            np.ndarray: The frame with particles drawn on it.
        """
    for p_f in particle_filters:
        for i in range(len(p_f.particles)):

            color = (p_f.weights[i] / p_f.max_weight) * 255
            centre = int(p_f.particles[i][0]), int(p_f.particles[i][1])
            if not grayscale:
                centre = centre[0] * 2, centre[1] * 2

            cv2.circle(frame, centre, 1, (255 - color, 0, color), 2)
    return frame


def show_particle_centre(frame: np.ndarray, particle_filters: List[ParticleFilter], grayscale: bool) -> np.ndarray:
    """
        Draws the center of particle filters.

        Args:
            frame (np.ndarray): The input image.
            particle_filters (List[ParticleFilter]): List of ParticleFilters.
            grayscale (bool): Boolean representing, if the image is grayscale.

        Returns:
            np.ndarray: The frame with particle center
    """

    for p_f in particle_filters:
        center = p_f.x, p_f.y
        if not grayscale:
            center = center[0] * 2, center[1] * 2
        cv2.circle(frame, center, 7, (0, 0, 255), 4)
    return frame


def show_ground_truth_location(frame: np.ndarray, frame_number: int, evaluator: SyntheticEvaluator,
                               grayscale: bool) -> None:
    """
        Draws ground truth annotations with ID

        Parameters:
            frame (np.ndarray): The input image.
            frame_number (int): The current frame number.
            evaluator (SyntheticEvaluator): An evaluator containing ground truth data for all frames.
            grayscale (bool):Boolean representing, if the image is grayscale.

        """
    if frame_number in evaluator.ground_truth_data:
        for annotation in evaluator.ground_truth_data[frame_number]:
            x, y = annotation["x"], annotation["y"]
            if not grayscale:
                x = x * 2
                y = y * 2
            cv2.circle(frame, (x , y), 5, (0, 0, 255), -1)  # Mark the point
            cv2.putText(frame, f'ID: {annotation["id"]}', (x + 10, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1)

    cv2.line(frame, (622, 0), (622, frame.shape[0]), (0, 255, 255), 2)