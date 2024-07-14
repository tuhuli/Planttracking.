from math import exp, radians, cos, sin, sqrt

import numpy as np
import scipy
from filterpy.monte_carlo import systematic_resample, multinomial_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import matplotlib.pyplot as plt


class ParticleFilter:
    def __init__(self):
        self.particles = None

    def predict(self, std):

        """ move according to control input u (velocity x, velocity y)
        with noise Q (std heading change, std velocity)`"""
        N = len(self.particles)

        # move in the (noisy) commanded direction
        self.particles[:, 0] += self.particles[:, 2] + (np.random.randn(N) * std[1])
        self.particles[:, 1] += self.particles[:, 3] + (np.random.randn(N) * std[1])

    def update(self, weights, last_position):
        for i, particle in enumerate(self.particles):

            # Compute the distance between each particle and the last known position
            dist = np.linalg.norm(particle[:2] - last_position)

            # Assign weight based on inverse distance (closer particles get higher weight)
            measurement_likelihood = scipy.stats.norm(dist, 1).pdf(0)

            # Update particle weight by multiplying with the measurement likelihood
            weights[i] *= measurement_likelihood

        weights += 1.e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize

    def update_with_velocity(self, weights, last_position, velocity, dist_coef=1, vel_coef=1):
        for i, particle in enumerate(self.particles):
            dist = np.linalg.norm(particle[:2] - last_position)
            vel_diff = velocity - particle[2:]

            measurement_likelihood = scipy.stats.norm(dist, 1).pdf(0)
            vel_likelihood_x = scipy.stats.norm(vel_diff[0], 1).pdf(0)
            vel_likelihood_y = scipy.stats.norm(vel_diff[1], 1).pdf(0)

# print(str(measurement_likelihood) + ": mes   |" + str(vel_likelihood_x) + ": vel_x")
            weights[i] *= measurement_likelihood**2 * dist_coef + (vel_likelihood_y + vel_likelihood_x)**2 * vel_coef

        weights += 1.e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize



    def estimate(self, weights):

        """returns mean and variance of the weighted particles"""
        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=weights, axis=0)
        var = np.average((pos - mean) ** 2,weights=weights, axis=0)
        return mean, var

    def resample_from_index(self, weights, indexes):
        """ resamplex the weigths from indexes, then set each weight to 1/N
            N - number of weight/particles"""
        self.particles[:] = self.particles[indexes]
        weights.resize(len(self.particles))
        weights.fill(1.0 / len(weights))

    def basic_resample_weights(self, weights):
        max_weight = weights.argmax()
        self.particles = create_gaussian_particles(self.particles[max_weight], (1,1,2,2), len(self.particles))
        weights.fill(1.0 / len(weights))

    def resample_with_last_measurment(self, weights, robot_velocity):
        max_weight = weights.argmax()
        std = robot_velocity[0] * 0.5, robot_velocity[1] * 0.5, robot_velocity[0] * 2, robot_velocity[1] * 2
        self.particles = create_gaussian_particles(self.particles[max_weight], std, len(self.particles))
        weights.fill(1.0 / len(weights))


def neff(weights):
    return 1. / np.sum(np.square(weights))


def create_uniform_particles(x_range, y_range, velocity_range, N):
    particles = np.empty((N, 4))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(velocity_range[0], velocity_range[1], size=N)
    particles[:, 3] = np.random.uniform(velocity_range[0], velocity_range[1], size=N)
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 4))

    particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
    particles[:, 3] = mean[3] + (np.random.randn(N) * std[3])
    return particles


def my_resample(weights):
    pass


def run_pf1(N, iters=40, sensor_std_err=.1, do_plot=True, plot_particles=True,xlim=(0, 50), ylim=(0, 50), initial_x=None):
    plt.figure()
    pf = ParticleFilter()
    pf.particles = create_uniform_particles((0, 20), (0, 20), (0.0, 5), N)

    weights = np.ones(N) / N

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)
        plt.scatter(pf.particles[:, 0], pf.particles[:, 1], alpha=alpha, color='g')

    xs = []
    robot_pos = np.array([0., 0.])
    for x in range(iters):
        robot_pos = robot_pos[0] + 1, robot_pos[1] +1

        # move diagonally forward to by speed of the particle
        pf.predict((0.2, 0.05))

        # incorporate measurements
        pf.update(weights,  robot_pos)

        # resample if too few effective particles
        if neff(weights) < N:
            indexes = multinomial_resample(weights)
            pf.resample_from_index(weights, indexes)
            assert np.allclose(weights, 1/N)

        mu, var = pf.estimate(weights)
        xs.append(mu)
        if plot_particles:
            plt.scatter(pf.particles[:, 0], pf.particles[:, 1],
                        color='k', marker=',', s=1)

        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',

                         color='k', s=180, lw=3)

        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()


def test_velocity_changes(iterations, show_test_points=False):
    x_values = []
    y_values = []
    x= 0
    y= 0
    dtx = 1
    dty = 1
    for i in range(iterations ):
        x+= dtx
        y += dty
        if i % 20 < 10:
            dtx += 1.3
            dty += 1
        else:
            dtx -= 1
            dty -= 1
        x_values.append(x)
        y_values.append(y)

    p1 = plt.scatter(x_values, y_values)
    if show_test_points:
        plt.show()
    return x_values, y_values


def test_velocity_changes_with_const_end(iterations):
    x_values, y_values = test_velocity_changes(iterations - 10)
    last_points = x_values[-1], y_values[-1]
    velocity = last_points[0] - x_values[-2], last_points[1] - y_values[-2]
    for i in range(10):
        x_values.append(x_values[-1] + velocity[0])
        y_values.append(y_values[-1] + velocity[1])
    return x_values, y_values



def run_pf_velocity_change(N, iters=40, sensor_std_err=.1, do_plot=True, plot_particles=True,xlim=(0, 900), ylim=(0, 300),
                             initial_x=None, dist_coef=1, vel_coef=1):
    plt.figure()
    pf = ParticleFilter()
    pf.particles = create_uniform_particles((0, 3), (0, 3), (0.50, 5), N)

    weights = np.ones(N) / N

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
        if (x > 1):
            pf.update_with_velocity(weights, robot_pos,
                                    robot_velocity,
                                    dist_coef=dist_coef, vel_coef=vel_coef)
        else:
            pf.update(weights, robot_pos)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            # indexes = multinomial_resample(weights)
            # pf.resample_from_index(weights, indexes)
            pf.resample_with_last_measurment(weights, robot_velocity)
            resamples_count += 1
            color_of_estimate = "b"
            assert np.allclose(weights, 1 / N)

        mu, var = pf.estimate(weights)
        xs.append(mu)
        if plot_particles:
            plt.scatter(pf.particles[:, 0], pf.particles[:, 1],
                        color='k', marker=',', s=1)

        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+', color='k', s=180, lw=3)

        p2 = plt.scatter(mu[0], mu[1], marker='s', color=color_of_estimate)

    xs = np.array(xs)
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.title("dist_coef: " + str(dist_coef) + "  | vel_coef: " + str(vel_coef))
    print('final position error, variance:\n\t', mu - np.array(robot_pos), var)
    print('resample took place: ', resamples_count)
    # plt.savefig("C:\SchoolApps\Bakalarka\Bakalarka_kod\graphs\dist_" + str(dist_coef) +"_vel_coef_"+ str(vel_coef))
    plt.show()



run_pf_velocity_change(3000, dist_coef=5, plot_particles=False)
