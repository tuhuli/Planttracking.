from typing import List, Tuple

import cv2
import numpy as np

import kalman_filter
from evaluation import Eval_class
from object_detection import get_plants_and_initialize_filter
from trackedObject import TrackedObject
from visualization import line_from_filter


def pair_filter_with_plants(filters: List[kalman_filter.KalmanFilterID],
                            plants: List[TrackedObject]) -> List[Tuple[kalman_filter.KalmanFilterID, TrackedObject]]:
    """
       Pairs each Kalman filter with the closest detected plant.

       Parameters:
           filters (List[kalman_filter.KalmanFilterID]): List of active Kalman filters.
           plants (List[TrackedObject]): List of detected plant objects.

       Returns:
           List[Tuple[kalman_filter.KalmanFilterID, TrackedObject]]: List of tuples where each tuple contains a 
           Kalman filter and its closest plant.
       """
    filters_with_closest_plants = []
    for k_filter in filters:
        k_filter.predict()

        closest_plant = None
        closest_distance = None
        for plant in plants:
            distance = abs(plant.x - k_filter.x[0])

            if distance < 100 and (closest_plant is None or closest_distance > distance):
                closest_plant = plant
                closest_distance = distance

        if closest_plant is None:
            k_filter.frames_not_found += 1
        else:
            k_filter.frames_not_found = 0
        filters_with_closest_plants.append((k_filter, closest_plant))

    return filters_with_closest_plants


def remove_duplicate_filters(filters: List[kalman_filter.KalmanFilterID]) -> None:
    """
    Removes duplicate Kalman filters that are too close to each other.

    Parameters:
        filters (List[kalman_filter.KalmanFilterID]): List of active Kalman filters.
    """

    for k_filter in filters:
        for k_filter2 in filters:
            if k_filter == k_filter2:
                continue
            if abs(k_filter.x[0] - k_filter2.x[0]) < 50:
                filters.remove(k_filter2)


def kalman_detection_on_video(new_cap: cv2.VideoCapture, read_cap: cv2.VideoCapture, out: cv2.VideoWriter,
                              grayscale: bool) -> None:
    """
    Applies Kalman filter-based object detection on a video and saves the output.

    Parameters:
        new_cap (cv2.VideoCapture): Capture of output video.
        read_cap (cv2.VideoCapture): Capture of input video.
        out (cv2.VideoWriter): writer to save the output video with bounding boxes.
        grayscale(bool) -> None
    """

    no_object_frames_counter = 10
    num_of_objects = 0
    filters: List[kalman_filter.KalmanFilterID] = []
    frame = 0

    evaluator = Eval_class("C:\SchoolApps\Bakalarka\Bakalarka_kod\ground_truth_data\ground_true_frames_SG19.json", 40)
    while True:
        out_ret, out_frame = new_cap.read()
        in_ret, in_frame = read_cap.read()

        if not out_ret or not in_ret:
            break

        threshold = 160
        thresh_image = np.where(in_frame < threshold, 0, in_frame)

        # Turn image to grayscale
        g_image = np.array(thresh_image[:, :, 0])
        plants, no_object_frames_counter, num_of_objects = get_plants_and_initialize_filter(g_image,
                                                                                            filters,
                                                                                            num_of_objects,
                                                                                            no_object_frames_counter,
                                                                                            'kalman')

        k_filter_plant_pair = pair_filter_with_plants(filters, plants)
        for k_filter, plant in k_filter_plant_pair:
            measurement = None

            if plant is not None:
                measurement = np.array([plant.x, plant.y, plant.bb_x_half, plant.bb_y_half])
                print(f"original = x:{plant.x} , y:{plant.y}")

            k_filter.update(measurement)

            if k_filter.x[0] > len(thresh_image[0]) or k_filter.x[0] < 0 or k_filter.frames_not_found > 15:
                filters.remove(k_filter)

            print("--------------")
            print(f"prediction {k_filter.id} = {k_filter.x} \n\n\n")
            if measurement is None or (
                    abs(measurement[0] - k_filter.x[0]) > 20 or abs(measurement[1] - k_filter.x[1]) > 20 or abs(
                measurement[2] - k_filter.x[2]) > 20 or abs(measurement[3] - k_filter.x[3]) > 20):
                print()
                print(f" {k_filter.P} \n")
            evaluator.check_if_found_plant(k_filter, frame, g_image.shape[1])
            if evaluator.get_current_ground_truth_frame() == frame:
                print(f"HAS PLANT: {evaluator.get_current_ground_truth_frame()}")
                print(f"lower_boundry = {g_image.shape[1] // 2 - evaluator.variance}")
                print(f"upper_boundry = {g_image.shape[1] // 2 + evaluator.variance}")

        remove_duplicate_filters(filters)
        # show_bounding_boxes_in_frame(thresh_image, out_frame, plants, filters, out, grayscale)
        if grayscale:
            image = in_frame
        else:
            image = out_frame
        image = line_from_filter(image, filters, grayscale)
        out.write(image)

        frame += 1

    new_cap.release()
    read_cap.release()
    out.release()
    
    evaluator.print_results()

    cv2.destroyAllWindows()
