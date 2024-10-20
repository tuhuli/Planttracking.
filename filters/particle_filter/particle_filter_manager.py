import numpy as np
from filters.filter_manager import FilterManager
from filters.particle_filter.particle_filter import ParticleFilter, create_uniform_particles


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
        new_particle_filter.particles = create_uniform_particles((x, x + w ), (y, y + h),
                                                                 velocity_range_x, velocity_range_y,
                                                                 self.number_of_particles)

        new_particle_filter.weights = np.ones(self.number_of_particles) / self.number_of_particles

        self.filters.append(new_particle_filter)
        self.initialized_filter = new_particle_filter
        new_particle_filter.estimate()
        return new_particle_filter

    def process_one_frame(self, grayscale_image, frame, evaluator, _):
        for  p_f in self.filters:
            if p_f == self.initialized_filter:
                self.initialized_filter = None
                continue

            p_f.predict((1, 0.1, 0.3, 0.05))
            p_f.update_with_image(grayscale_image)
            # p_f.print_best_weights(300)

            mu, var = p_f.estimate()

            if p_f.neff() < self.number_of_particles / 2:
                print("nef==================================")
                p_f.improved_systematic_resample()
                # p_f.basic_resample_weights()
                # assert np.allclose(p_f.weights, 1 / p_f.number_of_particles)
            print(f"position: {mu},  max_weight = {p_f.max_weight}")

            evaluator.check_if_found_plant(p_f, frame, grayscale_image.shape[1])
            if evaluator.get_current_ground_truth_frame() == frame:
                print(f"HAS PLANT: {evaluator.get_current_ground_truth_frame()}")
                print(f"lower_boundry = {grayscale_image.shape[1] // 2 - evaluator.variance}")
                print(f"upper_boundry = {grayscale_image.shape[1] // 2 + evaluator.variance}")
