from typing import List, Tuple

import numpy as np
from filters.filter_manager import FilterManager
from filters.kalman_filter.kalman_filter import KalmanFilterID
from utilities.trackedObject import TrackedObject
from utilities.evaluation import SyntheticEvaluator
from filterpy.common import Q_discrete_white_noise


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

        f = KalmanFilterID(dim_x=4, dim_z=2, id=self.id_counter)
        f.x = np.array([o.x, o.y, 13., 0.])
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

    def process_one_frame(self,frame_number: int, grayscale_image, evaluator: SyntheticEvaluator, plants):
        k_filter_plant_pair = self.pair_filter_with_plants(plants)
        for k_filter, plant in k_filter_plant_pair:
            if k_filter == self.initialized_filter:
                self.initialized_filter = None
                continue

            measurement = None

            if plant is not None:
                measurement = np.array([plant.x, plant.y])
                #print(f"original = x:{plant.x} , y:{plant.y},")

            k_filter.update(measurement)
            evaluator.save_result(k_filter.id, frame_number, k_filter.x[0], k_filter.x[1])

#            if k_filter.id >= 110:
     #           k_filter.print_information()
    #        evaluator.check_if_found_plant(k_filter, frame, grayscale_image.shape[1])
    #        if evaluator.get_current_ground_truth_frame() == frame_number:
     #           print(f"HAS PLANT: {evaluator.get_current_ground_truth_frame()}")
     #           print(f"lower_boundary = {grayscale_image.shape[1] // 2 - evaluator.variance}")
     #          print(f"upper_boundary = {grayscale_image.shape[1] // 2 + evaluator.variance}")

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
