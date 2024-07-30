import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats


class ParticleFilter:
    """
    A class to represent a Particle Filter.
    """
    _id_counter = 0

    def __init__(self, max_x, max_y):
        self.particles = None
        self.weights = None
        self.max_width = max_x
        self.max_height = max_y
        self.x = None
        self.y = None
        self.number_of_particles = 1000
        self.id = ParticleFilter._id_counter
        ParticleFilter._id_counter += 1

    def predict(self, std):
        """
        Move particles according to control input with noise.

        Parameters:
            std (tuple): The standard deviation for the noise in movement.
        """
        N = len(self.particles)
        self.particles[:, 0] += self.particles[:, 2] + (np.random.randn(N) * std[1])
        self.particles[:, 1] += self.particles[:, 3] + (np.random.randn(N) * std[1])

    def update(self, last_position):
        """
        Update particle weights based on the last known position.

        Parameters:
            last_position (numpy.ndarray): The last known position.
        """
        for i, particle in enumerate(self.particles):
            # Compute the distance between each particle and the last known position
            dist = np.linalg.norm(particle[:2] - last_position)

            # Assign weight based on inverse distance (closer particles get higher weight)
            measurement_likelihood = scipy.stats.norm(dist, 1).pdf(0)

            # Update particle weight by multiplying with the measurement likelihood
            self.weights[i] *= measurement_likelihood

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def update_with_velocity(self, last_position, velocity, dist_coef=1, vel_coef=1):
        """
        Update particle weights based on the last known position and velocity.

        Parameters:
             last_position (numpy.ndarray): The last known position.
             velocity (numpy.ndarray): The velocity of the object.
             dist_coef (float): Coefficient for distance likelihood.
             vel_coef (float): Coefficient for velocity likelihood.

        """
        for i, particle in enumerate(self.particles):
            dist = np.linalg.norm(particle[:2] - last_position)
            vel_diff = velocity - particle[2:]

            measurement_likelihood = scipy.stats.norm(dist, 1).pdf(0)
            vel_likelihood_x = scipy.stats.norm(vel_diff[0], 1).pdf(0)
            vel_likelihood_y = scipy.stats.norm(vel_diff[1], 1).pdf(0)

            # print(str(measurement_likelihood) + ": mes   |" + str(vel_likelihood_x) + ": vel_x")
            self.weights[i] *= measurement_likelihood ** 2 * dist_coef + (
                    vel_likelihood_y + vel_likelihood_x) ** 2 * vel_coef

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def update_with_image(self, image):
        """
        Update particle weights based on the image likelihood.

        Parameters:
            image (numpy.ndarray): The image to update weights from.
        """
        for i, particle in enumerate(self.particles):
            if (int(particle[0]) >= self.max_width or int(particle[0]) < 0
                    or int(particle[1]) >= self.max_height or int(particle[1]) < 0):
                self.weights[i] = 0
            else:
                measurement_likelihood = image[int(particle[1]), int(particle[0])]
                self.weights[i] *= measurement_likelihood
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def estimate(self):
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

    def resample_from_index(self, indexes):
        """
        Resample the particles based on indexes.

        Parameters:
            indexes (numpy.ndarray): The indexes to resample from.
        """
        self.particles[:] = self.particles[indexes]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def basic_resample_weights(self):
        """
        Resample weights using the particle with the maximum weight.
        """
        max_weight = self.weights.argmax()
        self.particles = create_gaussian_particles(self.particles[max_weight], (2, 2, 5, 5), len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def resample_with_last_measurement(self, robot_velocity):
        """
        Resample weights using the particle with the maximum weight and last measurement.

        Parameters:
            robot_velocity (numpy.ndarray): The velocity of the robot.
        """
        max_weight = self.weights.argmax()
        std = robot_velocity[0] * 0.5, robot_velocity[1] * 0.5, robot_velocity[0] * 2, robot_velocity[1] * 2
        self.particles = create_gaussian_particles(self.particles[max_weight], std, len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def neff(self):
        """
        Return the effective number of particles.

        Returns:
            float: The effective number of particles.
        """
        return 1. / np.sum(np.square(self.weights))

    def print_best_weights(self, number_of_prints):
        """
        Print the top N weights.

        Parameters:
            number_of_prints (int): The number of top weights to print.
        """
        top_indices = np.argpartition(self.weights, -number_of_prints)[:-number_of_prints]
        best_weights = self.weights[top_indices]
        print(f"Best weights :{best_weights}")

    def convert_particles_to_int(self):
        """
            Convert particle positions to integers.
        """
        self.particles[:, 0] = self.particles[:, 0].astype(int)
        self.particles[:, 1] = self.particles[:, 1].astype(int)

    def get_bb(self, size):
        """
        Return the bounding box coordinates for the particle filter.

        Parameters:
            size (int): The size of the bounding box.

        Returns:
            tuple: The bounding box coordinates.
        """
        return ((self.x - size, self.y - size),
                (self.x + size, self.y - size),
                (self.x - size, self.y + size),
                (self.x + size, self.y + size))

    def get_color_bb(self):
        """
        Return the color bounding box coordinates for the particle filter.

        Returns:
            tuple: The color bounding box coordinates.
        """

        bb = self.get_bb(30)
        return ((bb[0][0] * 2, bb[0][1] * 2), (bb[1][0] * 2, bb[1][1] * 2),
                (bb[2][0] * 2, bb[2][1] * 2), (bb[3][0] * 2, bb[3][1] * 2))


def create_uniform_particles(x_range, y_range, velocity_range_x, velocity_range_y, N):
    """
    Create particles with a uniform distribution.

    Parameters:
        x_range (tuple): The range of x values.
        y_range (tuple): The range of y values.
        velocity_range_x (tuple): The range of x velocities.
        velocity_range_y (tuple): The range of y velocities.
        N (int): The number of particles.

    Returns:
        numpy.ndarray: The created particles.
    """

    particles = np.empty((N, 4))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(velocity_range_x[0], velocity_range_x[1], size=N)
    particles[:, 3] = np.random.uniform(velocity_range_y[0], velocity_range_y[1], size=N)
    return particles


def create_gaussian_particles(mean, std, N):
    """
    Create particles with a Gaussian distribution.

    Parameters:
        mean (tuple): The mean position and velocity (x, y, vx, vy).
        std (tuple): The standard deviation for position and velocity (sx, sy, svx, svy).
        N (int): The number of particles.

    Returns:
        numpy.ndarray: The created particles.
    """
    particles = np.empty((N, 4))

    particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
    particles[:, 3] = mean[3] + (np.random.randn(N) * std[3])
    return particles


def test_velocity_changes(iterations, show_test_points=False):
    """
    Simulate changes in velocity over time.

    Parameters:
        iterations (int): The number of iterations for the simulation.
        show_test_points (bool): Whether to show the test points plot.

    Returns:
        tuple: Lists of x and y values for each iteration.
    """
    x_values = []
    y_values = []
    x = 0
    y = 0
    dtx = 1
    dty = 1
    for i in range(iterations):
        x += dtx
        y += dty
        if i % 20 < 10:
            dtx += 1.3
            dty += 1
        else:
            dtx -= 1
            dty -= 1
        x_values.append(x)
        y_values.append(y)

    if show_test_points:
        plt.show()
    return x_values, y_values


def test_velocity_changes_with_const_end(iterations):
    """
    Simulate changes in velocity over time and then keep a constant velocity at the end.

    Parameters:
        iterations (int): The number of iterations for the simulation.

    Returns:
        tuple: Lists of x and y values for each iteration.
    """
    x_values, y_values = test_velocity_changes(iterations - 10)
    last_points = x_values[-1], y_values[-1]
    velocity = last_points[0] - x_values[-2], last_points[1] - y_values[-2]
    for i in range(10):
        x_values.append(x_values[-1] + velocity[0])
        y_values.append(y_values[-1] + velocity[1])
    return x_values, y_values


def run_pf_velocity_change(N, iters=40, sensor_std_err=.1, do_plot=True, plot_particles=True, x_lim=(0, 900),
                           ylim=(0, 300),
                           initial_x=None, dist_coef=1, vel_coef=1):
    plt.figure()
    pf = ParticleFilter()
    pf.particles = create_uniform_particles((0, 3), (0, 3), (0.50, 5), N)

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(pf.particles[:, 0], pf.particles[:, 1], alpha=alpha, color='g')

    xs = []
    # x_positions, y_positions = test_velocity_changes(iters)
    x_positions, y_positions = test_velocity_changes_with_const_end(iters)
    robot_pos = np.array([x_positions[0], y_positions[0]])

    resamples_count = 0

    for x in range(1, iters):
        color_of_estimate = "r"
        prev_robot_position = robot_pos
        robot_pos = np.array([x_positions[x], y_positions[x]])
        robot_velocity = robot_pos - prev_robot_position

        # move diagonally forward to (x+1, y+1)
        pf.predict((.2, .05))

        # incorporate measurements
        if x > 1:
            pf.update_with_velocity(robot_pos,
                                    robot_velocity,
                                    dist_coef=dist_coef, vel_coef=vel_coef)
        else:
            pf.update(robot_pos)

        # resample if too few effective particles
        if pf.neff() < N / 2:
            # indexes = multinomial_resample(weights)
            # pf.resample_from_index(weights, indexes)
            pf.resample_with_last_measurement(robot_velocity)
            resamples_count += 1
            color_of_estimate = "b"
            assert np.allclose(pf.weights, 1 / N)

        mu, var = pf.estimate()
        xs.append(mu)
        if plot_particles:
            plt.scatter(pf.particles[:, 0], pf.particles[:, 1],
                        color='k', marker=',', s=1)

        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+', color='k', s=180, lw=3)

        p2 = plt.scatter(mu[0], mu[1], marker='s', color=color_of_estimate)

    xs = np.array(xs)
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*x_lim)
    plt.ylim(*ylim)
    plt.title("dist_coef: " + str(dist_coef) + "  | vel_coef: " + str(vel_coef))
    print('final position error, variance:\n\t', mu - np.array(robot_pos), var)
    print('resample took place: ', resamples_count)
    # plt.savefig("C:\SchoolApps\Bakalarka\Bakalarka_kod\graphs\dist_" + str(dist_coef) +"_vel_coef_"+ str(vel_coef))
    plt.show()
