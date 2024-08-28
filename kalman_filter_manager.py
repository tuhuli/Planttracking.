import numpy as np

from filter_manager import FilterManager
from kalman_filter import KalmanFilterID
from trackedObject import TrackedObject


class KalmanFilterManager(FilterManager):
    def __init__(self):
        super().__init__()
        self.id_counter = 0

    def initialize_filter(self, o: TrackedObject) -> KalmanFilterID:
        """
        Initializes a Kalman filter for a given object.
    
        Parameters:
            o (TrackedObject): The tracked object to initialize the Kalman filter with.
            id (int): Unique identifier for the Kalman filter.
    
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
