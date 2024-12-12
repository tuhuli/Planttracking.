from typing import Tuple, Any

import cv2
import numpy as np
from filters.filter_manager import FilterManager
from filters.kalman_filter.kalman_filter_manager import KalmanFilterManager
from filters.particle_filter.particle_filter_manager import ParticleFilterManager
from skimage import io as skio


def get_plants_and_initialize_filter(image: np.ndarray,
                                     f_manager: FilterManager,
                                     no_object_frames_counter: int,
                                     ) -> Tuple[list[float, float], int]:
    """
        Creates a TrackedObject for each plant detected in an image and initializes filters.

        Parameters:
            image (numpy.ndarray): T input grayscale image.
            f_manager (List[kalman_filter.KalmanFilterID] | List[ParticleFilter]): List of active Kalman filters.
            no_object_frames_counter (int): Number of frames without detected plant in initialization area.

        Returns:
            List[TrackedObject]: List of detected objects.
            int: Updated number of detected objects.
            int: Updated number of frames without detected plant in initialization area.
        """
    plant_in_initialization_area = False
    plants = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)

    for i in range(0, num_labels):
        x, y, w, h, area, c_x, c_y = get_object_stats(i, stats, centroids)

        if 3000 < area < 10000:
            plants.append((x + w // 2, y + h // 2))

            if c_x < 150 and no_object_frames_counter > 5:
                plant_in_initialization_area = True
                height, width = image.shape
                f_manager.initialize_filter(width, height, x, y, w, h)

            elif c_x < 150:
                plant_in_initialization_area = True

    if plant_in_initialization_area:
        no_object_frames_counter = 0
    else:
        no_object_frames_counter += 1

    return plants, no_object_frames_counter


def get_object_stats(i: int, stats: np.ndarray, centroids: np.ndarray) -> tuple[
    np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[
        Any, Any], Any, Any]:
    """
    Extracts the statistics and centroid coordinates of a connected component.

    Parameters:
        i (int): The index of the connected component.
        stats (np.ndarray): The statistics of all connected components.
        centroids (np.ndarray): The centroid coordinates of all connected components.

    Returns:
        tuple[int, int, int, int, int, float, float]:
            - x (int): The leftmost (x) coordinate of the bounding box.
            - y (int): The topmost (y) coordinate of the bounding box.
            - w (int): The width of the bounding box.
            - h (int): The height of the bounding box.
            - area (int): The area (in pixels) of the connected component.
            - c_x (float): The x-coordinate of the centroid.
            - c_y (float): The y-coordinate of the centroid.
    """
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    area = stats[i, cv2.CC_STAT_AREA]
    c_x, c_y = centroids[i]

    return x, y, w, h, area, c_x, c_y




