from filterpy.kalman import KalmanFilter


class KalmanFilterID(KalmanFilter):
    """
        KalmanFilterID extends KalmanFilter with ID.
        """

    def __init__(self, dim_x: int, dim_z: int, filter_id: int):
        super().__init__(dim_x=dim_x, dim_z=dim_z)
        self.id = filter_id
        self.frames_not_found = 0

    def get_centre_x(self) -> float:
        """
            Returns the x-coordinate of the filter's estimated center.

            Returns:
                int: The x-coordinate of the filter's center.
        """
        return self.x[0]

    def get_centre_y(self) -> float :
        """
            Returns the y-coordinate of the filter's estimated center.

            Returns:
                int: The y-coordinate of the filter's center.
        """
        return self.x[1]
