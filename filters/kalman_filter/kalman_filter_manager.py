from typing import List, Tuple

import numpy as np
from filters.filter_manager import FilterManager
from filters.kalman_filter.kalman_filter import KalmanFilterID
from utilities.trackedObject import TrackedObject


class KalmanFilterManager(FilterManager):
    def __init__(self):
        super().__init__()
        self.id_counter = 0

    def initialize_filter(self, o: TrackedObject) -> KalmanFilterID:
        """
        Initializes a Kalman filter for a given object.
    
        Parameters:
            o (TrackedObject): The tracked object to initialize the Kalman filter with.
    
        Returns:
            KalmanFilterID: The initialized Kalman filter.
        """
        self.id_counter += 1

        f = KalmanFilterID(dim_x=8, dim_z=4, id=self.id_counter)
        f.x = np.array([o.x, o.y, o.bb_x_half, o.bb_y_half, 5., 0., 0., 0.])
        f.F = np.array([[1., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 1.],
                        [0., 0., 0., 0., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 1.]])

        f.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0.]])

        f.P *= np.array([[25., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 25., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 10., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 10., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 25., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 25., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 25., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 25.]])
        f.R = np.array([[10., 0., 0., 0.],
                        [0., 10., 0., 0.],
                        [0., 0., 10000000., 0.],
                        [0., 0., 0., 10000000.]])

        white_noise_matrix = np.random.uniform(0.05, 0.2, size=(8, 8))
        f.Q = white_noise_matrix
        return f

    def process_one_frame(self, grayscale_image, frame, evaluator, plants):
        k_filter_plant_pair = self.pair_filter_with_plants(plants)
        for k_filter, plant in k_filter_plant_pair:
            measurement = None

            if plant is not None:
                measurement = np.array([plant.x, plant.y, plant.bb_x_half, plant.bb_y_half])
                print(f"original = x:{plant.x} , y:{plant.y}")

            k_filter.update(measurement)

            if k_filter.x[0] > len(grayscale_image[0]) or k_filter.x[0] < 0 or k_filter.frames_not_found > 15:
                self.filters.remove(k_filter)

            print("--------------")
            print(f"prediction {k_filter.id} = {k_filter.x} \n\n\n")

            evaluator.check_if_found_plant(k_filter, frame, grayscale_image.shape[1])
            if evaluator.get_current_ground_truth_frame() == frame:
                print(f"HAS PLANT: {evaluator.get_current_ground_truth_frame()}")
                print(f"lower_boundary = {grayscale_image.shape[1] // 2 - evaluator.variance}")
                print(f"upper_boundary = {grayscale_image.shape[1] // 2 + evaluator.variance}")

    def pair_filter_with_plants(self, plants: List[TrackedObject]) -> List[Tuple[KalmanFilterID, TrackedObject]]:
        """
           Pairs each Kalman filter with the closest detected plant.
    
           Parameters:
               plants (List[TrackedObject]): List of detected plant objects.
    
           Returns:
               List[Tuple[kalman_filter.KalmanFilterID, TrackedObject]]: List of tuples where each tuple contains a 
               Kalman filter and its closest plant.
           """
        filters_with_closest_plants = []
        for k_filter in self.filters:
            k_filter.predict()

            closest_plant = None
            closest_distance = None
            for plant in plants:
                distance = abs(plant.x - k_filter.x[0])

                if distance < 100 and (closest_plant is None or closest_distance > distance):
                    closest_plant = plant
                    closest_distance = distance

            if closest_plant is None:
                k_filter.frames_not_found += 1
            else:
                k_filter.frames_not_found = 0
            filters_with_closest_plants.append((k_filter, closest_plant))

        return filters_with_closest_plants
