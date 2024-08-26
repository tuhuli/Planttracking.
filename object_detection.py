from typing import Tuple

import cv2
import numpy as np
from skimage import io as skio

from filter_manager import FilterManager
from kalman_filter import initialize_kalman_filter
from trackedObject import TrackedObject


def get_plants_and_initialize_filter(image: np.ndarray,
                                     f_manager: FilterManager,
                                     num_of_objects: int,
                                     no_object_frames_counter: int,
                                     filter: str) -> Tuple[list[TrackedObject], int, int]:
    """
        Creates a TrackedObject for each plant detected in an image. If no plant is in initialize area, initialize a 
        filter. Detected plant is a connected component with area of 3000 to 10000 pixels.

        Parameters:
            image (numpy.ndarray): The input grayscale image.
            f_manager (List[kalman_filter.KalmanFilterID] | List[ParticleFilter]): List of active Kalman filters.
            num_of_objects (int): Number of detected objects to use as ID for new Kalman filter.
            no_object_frames_counter (int): Number of frames without detected plant in initialization area.
            filter (str): Name of the filter to initialize. 'particle'/'kalman'

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
            tr_object = TrackedObject(x + w // 2, y + h // 2, w // 2, h // 2, )

            plants.append(tr_object)

            if c_x < 150 and no_object_frames_counter > 3:
                plant_in_initialization_area = True
                num_of_objects += 1

                if filter == "kalman":
                    k_f = initialize_kalman_filter(tr_object, num_of_objects)
                    # filters.append(k_f)

                elif filter == "particle":
                    height, width = image.shape
                    f_manager.initialize_filter(max_width=width, max_height=height, x=x, y=y, w=w, h=h)


            elif c_x < 150:
                plant_in_initialization_area = True

    if plant_in_initialization_area:
        no_object_frames_counter = 0
    else:
        no_object_frames_counter += 1

    return plants, no_object_frames_counter, num_of_objects


def get_object_stats(i: int, stats: np.ndarray, centroids: np.ndarray) -> Tuple[int, int, int, int, int, float, float]:
    """
    Extracts the statistics and centroid coordinates of a connected component.

    Parameters:
        i (int): The index of the connected component.
        stats (numpy.ndarray): The statistics of all connected components. This array is typically obtained
                               from the `cv2.connectedComponentsWithStats` function.
        centroids (numpy.ndarray): The centroid coordinates of all connected components. This array is typically 
        obtained
                                   from the `cv2.connectedComponentsWithStats` function.

    Returns:
        Tuple[int, int, int, int, int, float, float]: 
            A tuple containing the following statistics and coordinates for the specified connected component:
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


def detect_objects(image, min_size=3000):
    numLabels, labels, stats, centroinds = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
    for i in range(0, numLabels):
        if i == 0:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]
        cX, cY = centroinds[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        print(f"INFO: Area = {area},   centroid= {centroinds[i]}")

    return image


def detection_test():
    image = cv2.imread("../Datasets/vineyard_screenshots/row_71_small/pred/predicted0169.png")
    image = np.array(image[:, :, 0])

    threshold = 200
    image = np.where(image < 200, 0, image)

    image = detect_objects(image[:, len(image[:]) - 200:])

    skio.imshow(image)
    skio.show()


if __name__ == "__main__":
    detection_test()
