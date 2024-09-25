class FilterManager:
    def __init__(self):
        self.filters = []

    def initialize_filter(self, **kwargs):
        pass

    def end_of_frame_cleanup(self):
        self.remove_duplicate_filters()
        self.remove_filter_outside_of_image()

    def remove_duplicate_filters(self):
        for p_filter in self.filters:
            for p_filter2 in self.filters:
                if p_filter == p_filter2:
                    continue
                if abs(p_filter.x - p_filter2.x) < 50:
                    self.filters.remove(p_filter2)

    def remove_filter_outside_of_image(self):
        for p_f in self.filters:
            if p_f.x > p_f.max_width - 40 or p_f.y > p_f.max_height or p_f.x < 0 or p_f.y < 0:
                self.filters.remove(p_f)

    def process_one_frame(self, grayscale_image, frame, evaluator, _):
        pass
