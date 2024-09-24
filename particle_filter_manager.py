import numpy as np

from filter_manager import FilterManager
from particle_filter import ParticleFilter
from particle_filter import create_uniform_particles


class ParticleFilterManager(FilterManager):
    def __init__(self):
        super().__init__()
        self.id_counter = 0
        self.number_of_particles = 1000

    def initialize_filter(self, max_width, max_height, x, y, w, h) -> ParticleFilter:
        self.id_counter += 1
        new_particle_filter = ParticleFilter(max_width, max_height, self.id_counter)

        if len(self.filters) == 0:
            velocity_range_x = (11, 15)

        else:
            velocity_range_x = (self.filters[-1].get_velocity_x() - 2, self.filters[-1].get_velocity_x() + 2)

        velocity_range_y = (-1, 1)
        new_particle_filter.particles = create_uniform_particles((x - w // 2, x + w // 2), (y - h // 2, y + h // 2),
                                                                 velocity_range_x, velocity_range_y,
                                                                 self.number_of_particles)

        new_particle_filter.weights = np.ones(self.number_of_particles) / self.number_of_particles

        self.filters.append(new_particle_filter)
        return new_particle_filter
