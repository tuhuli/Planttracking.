import cv2
import numpy as np

from evaluation import Eval_class
from object_detection import get_plants_and_initialize_filter
from particle_filter_manager import ParticleFilterManager
from visualization import show_particles_in_image, line_from_filter


def particle_detection_on_video(new_cap: cv2.VideoCapture, read_cap: cv2.VideoCapture, out: cv2.VideoWriter,
                                grayscale: bool, show_particles: bool) -> None:
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
    num_of_objects = 0
    particle_filter_manager = ParticleFilterManager()
    frame = 0
    evaluator = Eval_class("C:\SchoolApps\Bakalarka\Bakalarka_kod\ground_truth_data\ground_true_frames_SG19.json", 40)
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
        plants, no_object_frames_counter, num_of_objects = get_plants_and_initialize_filter(g_thresh_image,
                                                                                            particle_filter_manager,
                                                                                            num_of_objects,
                                                                                            no_object_frames_counter,
                                                                                            'particle')
        for p_f in particle_filter_manager.filters:
            p_f.predict((1, 0.1, 0.3, 0.05))
            p_f.convert_particles_to_int()
            p_f.update_with_image(g_image)
            # p_f.print_best_weights(300)

            mu, var = p_f.estimate()

            if p_f.neff() < particle_filter_manager.number_of_particles / 2:
                print("nef==================================")
                p_f.systematic_resample()
                # p_f.basic_resample_weights()
                # assert np.allclose(p_f.weights, 1 / p_f.number_of_particles)
            print(f"position: {mu},  max_weight = {p_f.max_weight}")

            evaluator.check_if_found_plant(p_f, frame, g_image.shape[1])
            if evaluator.get_current_ground_truth_frame() == frame:
                print(f"HAS PLANT: {evaluator.get_current_ground_truth_frame()}")
                print(f"lower_boundry = {g_image.shape[1] // 2 - evaluator.variance}")
                print(f"upper_boundry = {g_image.shape[1] // 2 + evaluator.variance}")

        particle_filter_manager.end_of_frame_cleanup()

        if grayscale:
            image = in_frame
        else:
            image = out_frame

        image = line_from_filter(image, particle_filter_manager.filters, grayscale)
        if show_particles:
            image = show_particles_in_image(image, particle_filter_manager.filters, grayscale)

        # io.imshow(image)
        # io.show()
        out.write(image)

        frame += 1

    evaluator.print_results()
