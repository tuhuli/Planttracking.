class FilterManager:
    def __init__(self):
        self.filters = []
        self.initialized_filter = None

    def initialize_filter(self, **kwargs):
        pass

    def end_of_frame_cleanup(self,height, width):
        self.remove_duplicate_filters()
        self.remove_filter_outside_of_image(height, width)

    def remove_duplicate_filters(self):
        for f in self.filters:
            for f2 in self.filters:
                if f == f2:
                    continue
                if abs(f.get_centre_x() - f2.get_centre_x()) < 50:
                    self.filters.remove(f2)

    def remove_filter_outside_of_image(self, height, width):
        for f in self.filters:
            if f.get_centre_x() > width - 40 or f.get_centre_y() > height or f.get_centre_x() < 0 or f.get_centre_y() < 0:
                self.filters.remove(f)

    def process_one_frame(self, frame_number, grayscale_image, evaluator, _):
        pass


