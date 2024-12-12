import json
from typing import List, Dict, Tuple

class SyntheticEvaluator:
    def __init__(self, path: str):
        self.filter_data = {}
        self.ground_truth_data: Dict[int, List[Dict[str, int]]] = load_ground_truth_data(path)
        self.id_switches: Dict[int, List[Tuple[int, int, int]]] = {}
        self.missed: Dict[int, List[Dict[str, int]]] = {}
        self.false_positive: Dict[int, List[Dict[str, int]]] = {}
        self.matches:Dict[int, List[Tuple[int, int, int]]] = {}
        self.total_distance: int = 0
        self.total_matches: int = 0
        self.total_false_positives: int = 0
        self.total_id_switches: int = 0
        self.total_missed: int = 0
        self.total_used_ground_truth: int = 0

    def match_filters_in_frame(self, frame_number:int):
        det_ids = set()
        for ground_truth_dic in self.ground_truth_data[frame_number]:
            if not is_in_evaluation_area(ground_truth_dic):
                continue
            self.total_used_ground_truth += 1

            closest_det = None
            closest_distance = None

            for detection_dic in self.filter_data[frame_number]:
                if not is_in_evaluation_area(detection_dic):
                    continue
                distance = abs(ground_truth_dic["x"] - detection_dic["x"])

                if distance < 50 and (closest_det is None or closest_distance > distance):
                    closest_det = detection_dic
                    closest_distance = distance

            # add to missing if the plant no match is made
            if closest_det is None:
                self.total_missed += 1
                self.missed[frame_number].append(ground_truth_dic)
                continue

            # make a match and check id switching
            self.total_matches += 1
            self.total_distance += closest_distance
            self.matches[frame_number].append((ground_truth_dic["id"], closest_det["id"], closest_distance))
            det_ids.add(closest_det["id"])

            if frame_number > 1:
                for ground_truth_id, detected_id, _ in self.matches[frame_number-1]:
                    if ground_truth_id == ground_truth_dic["id"] and closest_det["id"] != detected_id:
                        self.total_id_switches += 1
                        self.id_switches[frame_number].append((ground_truth_id, ground_truth_dic["id"], detected_id))

        # add false positive, if detection in not in matched
        for detection_dic in self.filter_data[frame_number]:
            if detection_dic["id"] not in det_ids and is_in_evaluation_area(detection_dic):
                self.total_false_positives += 1
                self.false_positive[frame_number].append(detection_dic)

    def evaluate(self) -> None:
        """
            Performs evaluation by comparing ground truth data with filter detections frame by frame.
        """
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

            current_frame += 1
        print("end_evaluation")

    def calculate_MOTP(self) -> None:
        """
            Calculates and prints the Multi-Object Tracking Precision (MOTP) metric.
        """
        MOTP = self.total_distance / self.total_matches
        print(f"Total matches = {self.total_matches}")
        print(f"Total distance = {self.total_distance}")
        print(f"MOTP: Total distance / Total matches = {MOTP}")

    def calculate_MOTA(self) -> None:
        """
            Calculates and prints the Multi-Object Tracking Accuracy (MOTA) metric for evaluation.
        """
        total_error = self.total_id_switches + self.total_missed + self.total_false_positives
        MOTA = 1 - total_error / self.total_used_ground_truth

        print(f"Total ID switches =  {self.total_id_switches}")
        print(f"Total false positives =  {self.total_false_positives}")
        print(f"Total missed detections = {self.total_missed}")
        print(f"Total errors (ID switches + false positive + missed detection) = {total_error}")
        print(f"Total ground truth = {self.total_used_ground_truth}")
        print(f"MOTA = f{MOTA}")

    def print_false_positives(self) -> None:
        """
            Prints the false positives for each frame and collects their unique IDs.
        """
        print("False positives:")
        ids = set()
        for frame, false_positives in self.false_positive.items():
            if false_positives:
                print(false_positives)
                print(frame)
        print(ids)

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


def is_in_evaluation_area(position_dict: Dict[str, int]) -> bool:
    """
        Checks if a given position is within the evaluation area.

        Parameters:
            position_dict (Dict[str, int]): A dictionary containing the position with an "x" key.

        Returns:
            bool: True if the position's "x" value is within the range [50, 622), otherwise False.
        """
    min_position = 50
    max_position = 622
    return max_position > position_dict["x"] >= min_position


def load_ground_truth_data(file_path: str) -> Dict[int, List[Dict[str, int]]]:
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