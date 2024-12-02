import json
from typing import List, Dict, Tuple


class SG_Evaluator:
    def __init__(self, path, variance):
        """
        Initializes the Evaluation class with two empty lists:
        - found_objects: Stores objects that were found.
        - not_found_objects: Stores objects that were not found.
        """
        self.found_objects = []
        self.not_found_objects = []
        self.ground_truth_list = self.load_sg_data(path)
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

    def load_sg_data(self,path: str) -> list[int]:
        with open(path, 'r') as file:
            data = json.load(file)
            return data




class SyntheticEvaluator():
    def __init__(self, path: str):
        self.filter_data = {}
        self.ground_truth_data: Dict[int, List[Dict[str, int]]] = self.load_ground_truth_data(path)
        self.id_switches: Dict[int, List[Tuple[int, int, int]]] = {}
        self.missed: Dict[int, List[Dict[str, int]]] = {}
        self.false_positive: Dict[int, List[Dict[str, int]]] = {}
        self.matches:Dict[int, List[Tuple[int, int, int]]] = {}
        self.total_distance: int = 0
        self.total_matches: int = 0
        self.total_false_positives: int = 0
        self.total_id_switches: int = 0
        self.total_missed: int = 0

    def match_filters_in_frame(self, frame_number:int):
        for ground_truth_dic in self.ground_truth_data[frame_number]:
            closest_det = None
            closest_distance = None

            for detection_dic in self.filter_data[frame_number]:
                distance = abs(ground_truth_dic["x"] - detection_dic["x"])

                if distance < 100 and (closest_det is None or closest_distance > distance):
                    closest_det = detection_dic
                    closest_distance = distance

            # add to missing if the plant no match is made
            if closest_det is None:
                self.total_missed += 1
                self.missed[frame_number].append(ground_truth_dic)
                continue

            #make a match and check id switching
            self.total_matches += 1
            self.total_distance += closest_distance
            self.matches[frame_number].append((ground_truth_dic["id"], closest_det["id"], closest_distance))

            if frame_number > 1:
                for ground_truth_id, detected_id, _ in self.matches[frame_number-1]:
                    if ground_truth_id == ground_truth_dic["id"] and closest_det["id"] != detected_id:
                        self.total_id_switches += 1
                        self.id_switches[frame_number].append((ground_truth_id, ground_truth_dic["id"], detected_id))


        # add false positive, if detection in not in matched
        det_ids = [match[1] for match in self.matches[0]]
        for detection_dic in self.filter_data[frame_number]:
            if detection_dic["id"] not in det_ids:
                self.total_false_positives += 1
                self.false_positive[frame_number].append(detection_dic)


    def evaluate(self):
        print("start evaluation")
        last_frame = max(max(self.ground_truth_data.keys()), max(self.filter_data.keys()))
        current_frame = 0

        while current_frame <= last_frame:
            self.matches[current_frame] = []
            self.missed[current_frame] = []
            self.false_positive[current_frame] = []
            self.id_switches[current_frame] = []

            if current_frame in self.ground_truth_data and current_frame in self.filter_data:
                self.match_filters_in_frame(current_frame)

            elif current_frame in self.ground_truth_data and current_frame not in self.filter_data:
                self.total_missed += 1
                self.missed[current_frame].append(self.ground_truth_data[current_frame])

            elif current_frame not in self.ground_truth_data and current_frame in self.filter_data:
                self.total_false_positives +=1
                self.false_positive[current_frame].extend(self.filter_data[current_frame])

            current_frame += 1
        print("end_evaluation")

    def calculate_MOTP(self):
        MOTP = self.total_distance / self.total_matches
        print(f"Total matches = {self.total_matches}")
        print(f"Total distance = {self.total_distance}")
        print(f"MOTP: Total distance / Total matches = {MOTP}")

    def calculate_MOTA(self):
        total_error = self.total_id_switches + self.total_missed + self.total_false_positives
        mota = 1 - total_error / len(self.ground_truth_data.values())


        print(f"Total ID switches =  {self.total_id_switches}")
        print(f"Total false positives =  {self.total_false_positives}")
        print(f"Total missed detections = {self.total_missed}")
        print(f"Total errors (ID switches + false positive + missed detection) = {total_error}")
        print(f"Total ground truth = {len(self.ground_truth_data.values())}")
        print(f"MOTA = f{mota}")

    def print_false_positives(self):

        print("False positives (only frames with false positives):")
        for frame, false_positives in self.false_positive.items():
            if false_positives:  # Check if there are any false positives in this frame
                print(false_positives)
                false_positive_ids = [fp["id"] for fp in false_positives]
                print(f"Frame {frame}: False Positive IDs: {false_positive_ids}")

    def save_result(self, object_id: int, frame: int, x: int, y: int) -> None :
        """
                Save the result data in the same format as ground truth data.

                Parameters:
                id (int): The unique identifier for the ellipse.
                frame (int): The frame number.
                position (Tuple[int, int]): The (x, y) position of the ellipse.
        """

        if frame not in self.filter_data:
            self.filter_data[frame] = []

        self.filter_data[frame].append({"id": object_id, "x": x, "y": y})

    def load_ground_truth_data(self, file_path: str) -> Dict[int, List[Dict[str, int]]]:
        """
            Load ground truth data from a JSON file.

            Parameters:
            file_path (str): The path to the JSON file containing ground truth data.

        """
        print("start loading gt_data")
        with open(file_path, "r") as file:
            ground_truth = json.load(file)

        ground_truth_dic = {int(frame_number): value for frame_number, value in ground_truth.items()}
        print("end loading gt_data")
        return ground_truth_dic

