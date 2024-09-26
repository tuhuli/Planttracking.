import json


class Evaluator:
    def __init__(self, path, variance):
        """
        Initializes the Evaluation class with two empty lists:
        - found_objects: Stores objects that were found.
        - not_found_objects: Stores objects that were not found.
        """
        self.found_objects = []
        self.not_found_objects = []
        self.ground_truth_list = load_data(path)
        self.ground_truth_index = 0
        self.variance = variance

    def is_in_centre(self, filter, image_width):
        image_centre = image_width // 2
        lower_boundary = image_centre - self.variance
        upper_boundary = image_centre + self.variance
        return lower_boundary < filter.get_centre_x() < upper_boundary

    def check_if_found_plant(self, filter, frame_index, image_width) -> None:
        if self.ground_truth_index > len(self.ground_truth_list):
            return

        if frame_index == self.get_current_ground_truth_frame() and self.is_in_centre(filter, image_width):
            self.ground_truth_index += 1
            self.found_objects.append((filter.id, frame_index))

        elif frame_index > self.get_current_ground_truth_frame():
            self.not_found_objects.append(frame_index)
            self.ground_truth_index += 1

    def get_current_ground_truth_frame(self):
        if self.ground_truth_index >= len(self.ground_truth_list):
            return -1
        return self.ground_truth_list[self.ground_truth_index]

    def print_results(self):
        """
               Prints the results in a descriptive format:
               1. Number of not found objects.
               2. List of not found objects.
               3. Number of found objects.
               4. List of found objects.
               """
        print("Number of not found objects:")
        print(len(self.not_found_objects))
        print("\nList of not found objects (frame indices):")
        print(self.not_found_objects)
        print("\nNumber of found objects:")
        print(len(self.found_objects))
        print("\nList of found objects (filter ID, frame index):")
        print(self.found_objects)


def load_data(path: str) -> list[int]:
    with open(path, 'r') as file:
        data = json.load(file)
        return data
