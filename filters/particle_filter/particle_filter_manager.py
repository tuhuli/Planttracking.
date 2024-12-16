import numpy as np
from filters.filter_manager import FilterManager
from filters.particle_filter.particle_filter import ParticleFilter, create_uniform_particles
from utilities.evaluation import SyntheticEvaluator


class ParticleFilterManager(FilterManager):
    """
        Manages a collection of particle filters for object tracking
    """
    def __init__(self):
        super().__init__()
        self.id_counter = 0
        self.number_of_particles = 500

    def initialize_filter(self, max_width: int, max_height: int, x: int, y: int, w: int, h: int) -> ParticleFilter:
        """
            Initializes a particle filter for a detected object.

            Parameters:
                max_width (int): Maximum width of the frame.
                max_height (int): Maximum height of the frame.
                x (int): X-coordinate of the object's bounding box.
                y (int): Y-coordinate of the object's bounding box.
                w (int): Width of the bounding box.
                h (int): Height of the bounding box.

            Returns:
                ParticleFilter: The initialized particle filter.
            """
        self.id_counter += 1
        new_particle_filter = ParticleFilter(max_width, max_height, self.id_counter)

        if len(self.filters) == 0:
            velocity_range_x = (5, 15)

        else:
            velocity_range_x = (self.filters[-1].get_velocity_x() - 2, self.filters[-1].get_velocity_x() + 2)

        velocity_range_y = (-1, 1)
        new_particle_filter.particles = create_uniform_particles((x, x + w), (y, y + h),
                                                                 velocity_range_x, velocity_range_y,
                                                                 self.number_of_particles)

        new_particle_filter.weights = np.ones(self.number_of_particles) / self.number_of_particles

        self.filters.append(new_particle_filter)
        self.initialized_filter = new_particle_filter
        new_particle_filter.estimate()
        return new_particle_filter

    def process_one_frame(self, frame_number: int, grayscale_image: np.ndarray, evaluator: SyntheticEvaluator,
                          _) -> None:
        """
            Processes a single frame by predicting, updating, and resampling particle filters.

            Parameters:
                frame_number (int): The current frame number.
                grayscale_image (np.ndarray): The grayscale image.
                evaluator (SyntheticEvaluator): Evaluator to save tracking results.
                _: Placeholder for additional parameters (unused).
        """
        for p_f in self.filters:
            if p_f == self.initialized_filter:
                self.initialized_filter = None
                continue

            p_f.predict((1, 0.2, 1, 0.05))
            p_f.update_with_image(grayscale_image)

            p_f.estimate()
            if evaluator is not None:
                evaluator.save_result(p_f.id, frame_number, p_f.x, p_f.y)

            if p_f.neff() < self.number_of_particles / 2:
                p_f.improved_systematic_resample()
