from typing import Tuple
import numpy as np
from utilities.utility_functions import bilinear_interpolate


class ParticleFilter:
    """
    A class to represent a particle Filter.
    """

    def __init__(self, max_x: int, max_y: int, id: int):
        self.particles = None
        self.weights = None
        self.max_width = max_x
        self.max_height = max_y
        self.x = None
        self.y = None
        self.id = id
        self.max_weight = 0

    def predict(self, std: Tuple[float, float]) -> None:
        """
        Move particles according to control input with added noise.

        Parameters:
            std (tuple[float, float]):
                A tuple representing the standard deviation of the noise:
        """
        N = len(self.particles)
        self.particles[:, 0] += self.particles[:, 2] + (np.random.randn(N) * std[0])
        self.particles[:, 1] += self.particles[:, 3] + (np.random.randn(N) * std[1])
        self.particles[:, 2] += np.random.randn(N) * std[2]
        self.particles[:, 3] += np.random.randn(N) * std[3]

    def update_with_image(self, image: np.ndarray) -> None:
        """
        Update particle weights based on the image likelihood.

        Parameters:
            image (numpy.ndarray): The image to update weights from.
        """
        for i, particle in enumerate(self.particles):
            x = particle[0]
            y = particle[1]

            if int(x) >= self.max_width or int(x) < 0 or int(y) >= self.max_height or int(y) < 0:
                self.weights[i] = 0
            else:
                measurement_likelihood = bilinear_interpolate(image, x, y)
                self.weights[i] = (self.weights[i] + 0.0005) * measurement_likelihood

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize
        self.max_weight = max(self.weights)

    def estimate(self) -> Tuple[float, float]:
        """
            Return the mean and variance of the weighted particles.

            Returns:
                tuple: mean and variance of the particles.
        """
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=self.weights, axis=0)
        self.x = int(mean[0])
        self.y = int(mean[1])
        return mean, var

    def systematic_resample(self) -> None:
        """
            Perform systematic resampling.
        """

        positions = (np.arange(len(self.particles)) + np.random.uniform(0, 1)) / len(self.particles)

        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0

        indices = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / len(self.weights))

    def improved_systematic_resample(self) -> None:
        """
            Perform  ISR.
        """

        relowering_threshold = 0.0001
        relowering_value = 1e-10

        self.weights[np.where(self.weights < relowering_threshold)] = relowering_value

        positions = (np.arange(len(self.particles)) + np.random.uniform(0, 1)) / len(self.particles)

        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0

        indices = np.searchsorted(cumulative_sum, positions)
        self.particles = self.particles[indices]
        # self.weights.fill(1.0 / len(self.weights))

    def neff(self) -> float:
        """
        Return the effective number of particles.

        Returns:
            float: The effective number of particles.
        """
        return 1. / np.sum(np.square(self.weights))

    def print_best_weights(self, number_of_prints: int) -> None:
        """
            Print the top N weights.

            Parameters:
                number_of_prints (int): The number of top weights to print.
        """
        top_indices = np.argpartition(self.weights, -number_of_prints)[:-number_of_prints]
        best_weights = self.weights[top_indices]
        print(f"Best weights :{best_weights}")

    def get_centre_x(self) -> float:
        """
            Returns the x-coordinate of the filter's estimated center.

            Returns:
                int: The x-coordinate of the filter's center.
        """
        return self.x

    def get_centre_y(self) -> float:
        """
            Returns the y-coordinate of the filter's estimated center.

            Returns:
                int: The y-coordinate of the filter's center.
        """
        return self.y

    def get_velocity_x(self) -> float:
        """
            Returns the average x-velocity of the particles, weighted by their weights.

            Returns:
                float: The weighted average x-velocity of the particles.
        """
        pos = self.particles[:, 2]
        mean_velocity = np.average(pos, weights=self.weights, axis=0)
        return mean_velocity


def create_uniform_particles(x_range: Tuple[int, int], y_range: Tuple[int, int],
                             velocity_range_x: Tuple[float, float],
                             velocity_range_y: Tuple[float, float],
                             N: int) -> np.ndarray:
    """
        Creates particles with uniform distributions .

        Parameters:
            x_range (Tuple[int, int]): The range of x positions as (min, max).
            y_range (Tuple[int, int]): The range of y positions as (min, max).
            velocity_range_x (Tuple[float, float]): The range of x velocities as (min, max).
            velocity_range_y (Tuple[float, float]): The range of y velocities as (min, max).
            N (int): The number of particles.

        Returns:
            np.ndarray: An array of shape (N, 4) where each particle has the form (x, y, vx, vy).

    """

    particles = np.empty((N, 4))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(velocity_range_x[0], velocity_range_x[1], size=N)
    particles[:, 3] = np.random.uniform(velocity_range_y[0], velocity_range_y[1], size=N)
    return particles


def create_gaussian_particles(mean: Tuple[float, float, float, float],
                              std: Tuple[float, float, float, float],
                              N: int) -> np.ndarray:
    """
        Creates particles with Gaussian distributions.

        Parameters:
            mean (Tuple[float, float, float, float]): The mean values for position (x, y) and velocity (vx, vy).
            std (Tuple[float, float, float, float]): The standard deviations for position (x, y) and velocity (vx, vy).
            N (int): The number of particles.

        Returns:
            np.ndarray: An array of shape (N, 4) where each particle has the form (x, y, vx, vy).
    """
    particles = np.empty((N, 4))

    particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (abs(np.random.randn(N)) * std[2])
    particles[:, 3] = mean[3] + (np.random.randn(N) * std[3])
    return particles
