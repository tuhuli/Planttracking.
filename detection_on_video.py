import cv2
import numpy as np

from filters.kalman_filter.kalman_filter_manager import KalmanFilterManager
from filters.particle_filter.particle_filter_manager import ParticleFilterManager
from utilities.evaluation import SyntheticEvaluator
from utilities.object_detection import get_plants_and_initialize_filter
from utilities.visualization import show_particles_in_image, line_from_filter, show_ground_truth_location


def tracking_on_video(new_cap: cv2.VideoCapture, read_cap: cv2.VideoCapture, out: cv2.VideoWriter,
                       grayscale: bool, show_particles: bool, filter_type: str) -> None:
    """
    Tracks objects in a video using either Particle or Kalman filters and saves the output.

    Parameters:
        new_cap (cv2.VideoCapture): Video capture for the output video.
        read_cap (cv2.VideoCapture): Video capture for the input video.
        out (cv2.VideoWriter): Writer to save the output video .
        grayscale (bool): Boolean representing, if the output video is grayscale.
        show_particles (bool): Boolean representing if the particle positions are shown in the output.
        filter_type (str): Type of filter to use ("particle" or "kalman").
    """
    no_object_frames_counter = 10
    frame_number = 0
    evaluator = SyntheticEvaluator(".\data\ground_truth_data\SG_annotations.json")

    if filter_type == "particle":
        filter_manager = ParticleFilterManager()
    else:
        filter_manager = KalmanFilterManager()

    while True:
        out_ret, out_frame = new_cap.read()
        in_ret, in_frame = read_cap.read()

        if not out_ret or not in_ret:
            break

        threshold = 200
        thresh_image = np.where(in_frame < threshold, 0, in_frame)

        # Turn image to grayscale
        g_thresh_image = np.array(thresh_image[:, :, 0])
        g_image = np.array(in_frame[:, :, 0])

        plants, no_object_frames_counter = get_plants_and_initialize_filter(g_thresh_image,
                                                                            filter_manager,
                                                                            no_object_frames_counter)

        filter_manager.process_one_frame(frame_number, g_image, evaluator, plants)
        height, width = g_thresh_image.shape
        filter_manager.end_of_frame_cleanup(height, width)

        if grayscale:
            #image = thresh_image
            image = in_frame
        else:
            image = out_frame

        image = line_from_filter(image, filter_manager.filters, grayscale)
        show_ground_truth_location(image, frame_number, evaluator, grayscale)
        if show_particles:
            image = show_particles_in_image(image, filter_manager.filters, grayscale)

        out.write(image)

        frame_number += 1

    evaluator.evaluate()
    #evaluator.print_false_positives()
    evaluator.calculate_MOTA()
    evaluator.calculate_MOTP()
