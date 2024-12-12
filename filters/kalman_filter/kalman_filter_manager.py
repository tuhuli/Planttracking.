from typing import List, Tuple

import numpy as np
from filters.filter_manager import FilterManager
from filters.kalman_filter.kalman_filter import KalmanFilterID
from utilities.evaluation import SyntheticEvaluator


class KalmanFilterManager(FilterManager):
    """
        Manages a collection of Kalman filters for tracking objects in a sequence of frames.
    """
    def __init__(self):
        super().__init__()
        self.id_counter = 0

    def initialize_filter(self, max_width: int, max_height: int, x: int, y: int, w: int, h: int) -> KalmanFilterID:
        """
        Initializes a Kalman filter for an object.

        Parameters:
            max_width (int): Maximum width of the frame.
            max_height (int): Maximum height of the frame.
            x (int): X-coordinate of the object's top-left corner.
            y (int): Y-coordinate of the object's top-left corner.
            w (int): Width of the bounding box.
            h (int): Height of the bounding box.

        Returns:
            KalmanFilterID: The initialized Kalman filter with a unique ID.
        """
        self.id_counter += 1

        f = KalmanFilterID(4, 2, self.id_counter)
        f.x = np.array([x + w // 2, y + h // 2, 13., 0.])
        f.F = np.array([[1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

        f.H = np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.]])

        f.P *= np.array([[25., 0., 0., 0.],
                         [0., 25., 0., 0.],
                         [0., 0., 25., 0.],
                         [0., 0., 0., 25.]])

        f.R = np.array([[1000., 0.],
                        [0., 1000.]])

        f.Q = np.array([[25., 0., 0., 0.],
                         [0., 25., 0., 0.],
                         [0., 0., 25., 0.],
                         [0., 0., 0., 25.]])

        self.filters.append(f)
        self.initialized_filter = f
        return f

    def process_one_frame(self, frame_number: int, grayscale_image: np.ndarray, evaluator: SyntheticEvaluator,
                          plants: list[Tuple[int, int]]) -> None:
        """
            Processes a single frame by pairing Kalman filters with detected plants, updating filter states,
            and saving results to the evaluator.

            Parameters:
                frame_number (int): The current frame number.
                grayscale_image (np.ndarray): The grayscale image.
                evaluator (SyntheticEvaluator):  Evaluator to save tracking results.
                plants (List[Tuple[float, float]]): List of detected plants coordinates as (x, y) tuples.
        """
        k_filter_plant_pair = self.pair_filter_with_plants(plants)
        for k_filter, plant_tuple in k_filter_plant_pair:
            if k_filter == self.initialized_filter:
                self.initialized_filter = None
                continue

            measurement = None
            if plant_tuple is not None:
                measurement = np.array([plant_tuple[0], plant_tuple[1]])

            k_filter.update(measurement)
            evaluator.save_result(k_filter.id, frame_number, k_filter.x[0], k_filter.x[1])

    def pair_filter_with_plants(self, plants: List[Tuple[float, float]]) -> List[
        Tuple[KalmanFilterID, Tuple[float, float] | None]]:
        """
            Pairs each Kalman filter with the closest detected object within a distance threshold.

            Parameters:
                plants (List[Tuple[float, float]]): List of detected object coordinates as (x, y) tuples.

            Returns:
                List[Tuple[KalmanFilterID, Tuple[float, float] | None]]:
                    A list of tuples where each tuple contains:
                    - A Kalman filter (KalmanFilterID).
                    - The closest detected object as (x, y) or None if no match is found.
        """
        filters_with_closest_plants = []
        for k_filter in self.filters:
            k_filter.predict()

            closest_plant = None
            closest_distance = None
            for (plant_x, plant_y) in plants:
                distance = abs(plant_x - k_filter.x[0])

                if distance < 100 and (closest_plant is None or closest_distance > distance):
                    closest_plant = plant_x, plant_y
                    closest_distance = distance

            if closest_plant is None:
                k_filter.frames_not_found += 1
            else:
                k_filter.frames_not_found = 0
            filters_with_closest_plants.append((k_filter, closest_plant))

        return filters_with_closest_plants
