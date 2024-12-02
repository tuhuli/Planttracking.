from typing import Tuple, List

import numpy as np
from filterpy.kalman import KalmanFilter
from utilities.trackedObject import TrackedObject


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

    def get_centre_x(self):
        return self.x[0]

    def get_centre_y(self):
        return self.x[1]


    def print_information(self):
        print(f"ID: {self.id} | position: {self.get_centre_x()} {self.get_centre_y()} | velocity: {self.x[2]} {self.x[3]}")
