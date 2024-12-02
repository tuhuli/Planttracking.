import cv2
import numpy as np

from filters.kalman_filter.kalman_filter_manager import KalmanFilterManager
from filters.particle_filter.particle_filter_manager import ParticleFilterManager
from utilities.evaluation import SyntheticEvaluator, SG_Evaluator
from utilities.object_detection import get_plants_and_initialize_filter
from utilities.visualization import show_particles_in_image, line_from_filter


def detection_on_video(new_cap: cv2.VideoCapture, read_cap: cv2.VideoCapture, out: cv2.VideoWriter,
                       grayscale: bool, show_particles: bool, filter_type: str) -> None:
    """
    Applies Particle filter-based object detection on a video and saves the output.

    Parameters:
        new_cap (cv2.VideoCapture): Capture of output video.
        read_cap (cv2.VideoCapture): Capture of input video.
        out (cv2.VideoWriter): writer to save the output video with bounding boxes.
        grayscale(bool): boolean to signal grayscale video
        show_particles(bool): boolean to signal if the particles should be shown in the video
    """
    no_object_frames_counter = 10
    frame_number = 0
    #evaluator = Evaluator(".\data\ground_truth_data\ground_true_frames_SG19.json", 40)
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

        threshold = 180
        thresh_image = np.where(in_frame < threshold, 0, in_frame)

        # Turn image to grayscale
        g_thresh_image = np.array(thresh_image[:, :, 0])
        g_image = np.array(in_frame[:, :, 0])
        plants, no_object_frames_counter = get_plants_and_initialize_filter(g_thresh_image,
                                                                            filter_manager,
                                                                            no_object_frames_counter)

        filter_manager.process_one_frame(frame_number ,g_image, evaluator, plants)
        height, width = g_thresh_image.shape
        filter_manager.end_of_frame_cleanup(height, width)

        if grayscale:
            image = thresh_image
            #image = in_frame
        else:
            image = out_frame

        image = line_from_filter(image, filter_manager.filters, grayscale)
        if show_particles:
            image = show_particles_in_image(image, filter_manager.filters, grayscale)

        # io.imshow(image)
        # io.show()
        out.write(image)

        frame_number += 1

    evaluator.evaluate()
    evaluator.print_false_positives()
    evaluator.calculate_MOTA()
    evaluator.calculate_MOTP()
