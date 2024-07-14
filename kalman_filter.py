from typing import Tuple, List

import numpy as np
from filterpy.kalman import KalmanFilter

from trackedObject import TrackedObject


class KalmanFilterID(KalmanFilter):
    """
        KalmanFilterID extends KalmanFilter to include an ID and frame not found counter.

        Attributes:
            id (int): Unique identifier for the Kalman filter.
            frames_not_found (int): Counter for frames where the object was not found.
        """

    def __init__(self, dim_x: int, dim_z: int, id: int):
        """
        Initializes the KalmanFilterID object.

        Parameters:
            dim_x (int): Dimension of the state vector.
            dim_z (int): Dimension of the measurement vector.
            id (int): Unique identifier for the Kalman filter.
        """
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.id = id
        self.frames_not_found = 0

    def get_bb(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Gets the bounding box coordinates based on the current state.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]: 
            The coordinates of the bounding box.
        """
        return ((int(self.x[0] - self.x[2]), int(self.x[1] - self.x[3])),
                (int(self.x[0] + self.x[2]), int(self.x[1] - self.x[3])),
                (int(self.x[0]) - self.x[2], int(self.x[1] + self.x[3])),
                (int(self.x[0] + self.x[2]), int(self.x[1] + self.x[3])))

    def get_color_bb(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Gets the bounding box coordinates for a color image.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]: 
            The coordinates of the bounding box, scaled for a color image.
        """
        bb = self.get_bb()
        return ((bb[0][0] * 2, bb[0][1] * 2), (bb[1][0] * 2, bb[1][1] * 2),
                (bb[2][0] * 2, bb[2][1] * 2), (bb[3][0] * 2, bb[3][1] * 2))


def initialize_kalman_filter(o: TrackedObject, id: int) -> KalmanFilterID:
    """
    Initializes a Kalman filter for a given object.

    Parameters:
        o (TrackedObject): The tracked object to initialize the Kalman filter with.
        id (int): Unique identifier for the Kalman filter.

    Returns:
        KalmanFilterID: The initialized Kalman filter.
    """
    f = KalmanFilterID(dim_x=8, dim_z=4, id=id)
    f.x = np.array([o.x, o.y, o.bb_x_half, o.bb_y_half, 1., 0., 0., 0.])
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
                     [0., 0., 100., 0., 0., 0., 0, 0.],
                     [0., 0., 0., 100., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 225., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 225., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 225., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 225.]])
    f.R = np.eye(4) * 1000

    white_noise_matrix = np.random.uniform(0.05, 0.2, size=(8, 8))
    f.Q = white_noise_matrix
    return f


def kalman_tracking(tr_objects: List[TrackedObject], f: KalmanFilterID) -> None:
    """
    Applies Kalman filter tracking to a list of tracked objects.

    Parameters:
        tr_objects (List[TrackedObject]): List of tracked objects.
        f (KalmanFilterID): The Kalman filter to use for tracking.
    """
    for i in range(1, len(tr_objects)):
        o = tr_objects[i]
        z = np.array([o.x, o.y, o.bb_x_half, o.bb_y_half])
        f.predict()
        f.update(z)

        print("original = ")
        print(o.x)
        print(o.y)
        print("")
        print("--------------")
        print("")
        print("prediction = ")
        print(f.x)
