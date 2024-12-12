import numpy as np


class FilterManager:
    """
        Base class for managing tracking filters.
    """
    def __init__(self):
        self.filters = []
        self.initialized_filter = None

    def initialize_filter(self, max_width: int, max_height: int, x: int, y: int, w: int, h: int):
        """
            Initializes a new filter for an object.
            To be implemented by subclasses.

            Parameters:
                max_width (int): Maximum width of the image.
                max_height (int): Maximum height of the image.
                x (int): X-coordinate of the object's top-left corner.
                y (int): Y-coordinate of the object's top-left corner.
                w (int): Width of the bounding box.
                h (int): Height of the bounding box.
        """
        pass

    def end_of_frame_cleanup(self, height: int, width: int) -> None:
        """
            Performs cleanup by removing duplicate and out-of-bounds filters.

            Parameters:
                height (int): Height of the image.
                width (int): Width of the image.


        """
        self.remove_duplicate_filters()
        self.remove_filter_outside_of_image(height, width)

    def remove_duplicate_filters(self) -> None:
        """
        Removes duplicate filters that are within 50 pixels to each other.
        """
        for f in self.filters:
            for f2 in self.filters:
                if f == f2:
                    continue
                if abs(f.get_centre_x() - f2.get_centre_x()) < 50:
                    self.filters.remove(f2)

    def remove_filter_outside_of_image(self, height: int, width: int) -> None:
        """
            Removes filters that are outside the image boundaries.

            Parameters:
                height (int): Height of the image.
                width (int): Width of the image.
        """
        for f in self.filters:
            if f.get_centre_x() > width - 40 or f.get_centre_y() > height or f.get_centre_x() < 0 or f.get_centre_y() < 0:
                self.filters.remove(f)

    def process_one_frame(self, frame_number: int, grayscale_image: np.ndarray, evaluator, _) -> None:
        """
            Processes a single frame.
            To be implemented by subclasses.

            Parameters:
                frame_number (int): The current frame number.
                grayscale_image (np.ndarray): The grayscale image.
                evaluator: Evaluator to save tracking results.
                _: Placeholder for additional parameters.
        """
        pass


