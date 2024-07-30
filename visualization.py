from typing import List

import cv2
import numpy as np

from kalman_filter import KalmanFilterID
from particle_filter import ParticleFilter
from preprocessing import initialize_tr
from trackedObject import TrackedObject


def visualize_bb(data, path, start_range=1, end_range=100):
    for frame_number in range(start_range, end_range):
        # Load an image
        image = cv2.imread(path + f"{frame_number:06d}.jpg")

        frame_data = data[data['Frame'] == frame_number]
        TR_objects = initialize_tr(frame_data)

        for tr_object in TR_objects:
            (pt1, _, _, pt2) = tr_object.get_bb()

            cv2.rectangle(image, int(pt1), int(pt2), (0, 255, 0), 2)

            text_size = cv2.getTextSize(tr_object.ID, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            font_scale = min(int(tr_object.bb_x_half * 2 / text_size[0]),
                             int(tr_object.bb_y_half * 2 / text_size[1])) / 1.5

            text_position = (int(pt1[0]), int(pt1[1] - 5))  # Adjust the position for the scaled text
            cv2.putText(image, tr_object.ID, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def bb_from_kalman_filter(image: np.ndarray, k_filters: List[KalmanFilterID],
                          grayscale: bool = True) -> np.ndarray:
    """
    Draws bounding boxes from Kalman filters on the given image.

    Parameters:
        image (np.ndarray): The input image on which bounding boxes are drawn.
        k_filters (List[KalmanFilterID]): List of active Kalman filters.
        grayscale (bool): Whether the image is grayscale. Default is True.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    for k_filter in k_filters:

        if grayscale:
            (pt1, _, _, pt2) = k_filter.get_bb()
        else:
            (pt1, _, _, pt2) = k_filter.get_color_bb()

        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)

        text_size = cv2.getTextSize(str(k_filter.id), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        font_scale = min(int(k_filter.x[2] * 2 / text_size[0]), int(k_filter.x[3] * 2 / text_size[1])) / 1.5

        # Write Object ID scaled with the bounding box size
        text_position = (int(pt1[0]), int(pt1[1] - 5))
        cv2.putText(image, str(k_filter.id), text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return image


def bb_from_tr_object(image: np.ndarray, tr_objects: List[TrackedObject],
                      grayscale: bool = True) -> np.ndarray:
    """
    Draws bounding boxes from tracked objects on the given image.

    Parameters:
        image (np.ndarray): The input image on which bounding boxes are drawn.
        tr_objects (List[TrackedObject]): List of tracked objects.
        grayscale (bool): Whether the image is grayscale. Default is True.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    for tr_object in tr_objects:
        if grayscale:
            (pt1, _, _, pt2) = tr_object.get_bb()
        else:
            (pt1, _, _, pt2) = tr_object.get_color_bb()

        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)

    return image


def bb_from_particle_filter(image: np.ndarray, p_filters: List[ParticleFilter],
                            grayscale: bool = True) -> np.ndarray:
    """
    Draws bounding boxes from particle filters on the given image.

    Parameters:
        image (np.ndarray): The input image on which bounding boxes are drawn.
        p_filters (List[object]): List of particle filters.
        grayscale (bool): Whether the image is grayscale. Default is True.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    for particle_filter in p_filters:
        if grayscale:
            (pt1, _, _, pt2) = particle_filter.get_bb(30)
        else:
            (pt1, _, _, pt2) = particle_filter.get_color_bb()

        cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)

        text_position = (int(pt1[0]), int(pt1[1] - 5))
        cv2.putText(image, str(particle_filter.id), text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    return image


def show_bounding_boxes_in_frame(frame: np.ndarray, color_frame: np.ndarray,
                                 plants: List[TrackedObject], filters: List[KalmanFilterID],
                                 out: cv2.VideoWriter, grayscale: bool) -> None:
    """
    Shows and writes bounding boxes in a video frame.

    Parameters:
        frame (np.ndarray): The grayscale frame image.
        color_frame (np.ndarray): The color frame image.
        plants (List[TrackedObject]): List of detected plant objects.
        filters (List[kalman_filter.KalmanFilterID]): List of active Kalman filters.
        out (cv2.VideoWriter): VideoWriter object to save the modified frame.
        grayscale (bool): Whether the frame is grayscale.

    """
    if grayscale:
        out_frame = frame
    else:
        out_frame = color_frame

    out_frame = bb_from_tr_object(out_frame, plants, grayscale)
    out_frame = bb_from_kalman_filter(out_frame, filters, grayscale)
    out.write(out_frame)


def show_particles_in_image():
    pass
