import argparse

import cv2

from kalman_filter_test import kalman_detection_on_video
from particle_filter_test import particle_detection_on_video


def use_filter_on_video(video_input_path: str, video_output_path: str,
                        color_video_path: str, filter: str) -> None:
    grayscale = color_video_path is None

    if grayscale:
        new_cap = cv2.VideoCapture(video_input_path)
    else:
        new_cap = cv2.VideoCapture(color_video_path)

    fps = new_cap.get(cv2.CAP_PROP_FPS)
    width = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    read_cap = cv2.VideoCapture(video_input_path)

    if filter == "particle":
        particle_detection_on_video(new_cap, read_cap, out, grayscale)
    elif filter == 'kalman':
        kalman_detection_on_video(new_cap, read_cap, out, grayscale)
    else:
        print("Wrong type of filter. Use: 'kalman'/'particle")

    new_cap.release()
    read_cap.release()
    out.release()

    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video and evaluate results.')

    # Required argument
    parser.add_argument("filter", type=str, help='Specify filter: kalman/particle')
    parser.add_argument('video_path', type=str, help='Path to the video file')

    # Optional arguments
    parser.add_argument('--original_video_path', type=str, help='Path to the original video file')
    parser.add_argument('--output_video_path', type=str,
                        help='Path to store the new created video file, if empty it will be stored in the current '
                             'directory')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.output_video_path is None:
        args.output_video_path = "./output_tracked_video.mp4"

    use_filter_on_video(args.video_path, args.output_video_path, args.original_video_path, args.filter)
