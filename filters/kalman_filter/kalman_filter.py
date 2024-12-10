from typing import Tuple, List

import numpy as np
from filterpy.kalman import KalmanFilter
from utilities.trackedObject import TrackedObject


class KalmanFilterID(KalmanFilter):
    """
        KalmanFilterID extends KalmanFilter with ID and frame_not_found counter.

        Attributes:
            id (int): ID for the Kalman filter.
            frames_not_found (int): Counter for frames where the object was not found.
        """

    def __init__(self, dim_x: int, dim_z: int, id: int):
        """
        Initializes the KalmanFilterID object.

        Parameters:
            dim_x (int): Dimension of the state vector.
            dim_z (int): Dimension of the measurement vector.
            id (int): ID for the Kalman filter.
        """
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.id = id
        self.frames_not_found = 0

    def get_centre_x(self):
        """
        Returns the x-coordinate of the filter position
        """
        return self.x[0]

    def get_centre_y(self):
        """
        Returns the y-coordinate of the filter position
        """
        return self.x[1]

    def print_information(self):
        """
        Prints information about filter.
        The informations are:
            ID
            Position
            Velocity
        """
        print(f"ID: {self.id} | position: {self.get_centre_x()} {self.get_centre_y()} | velocity: {self.x[2]} {self.x[3]}")
