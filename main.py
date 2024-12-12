import argparse
import cv2

from detection_on_video import tracking_on_video
from utilities.evaluation import SyntheticEvaluator


def use_filter_on_video(video_input_path: str, video_output_path: str,
                        color_video_path: str, filter_type: str, ground_truth_path: str | None, show_particles: bool) -> None:
    """
        Applies a tracking filter to a video and saves the processed output.

        Parameters:
            video_input_path (str): Path to the input video.
            video_output_path (str): Path to save the output video.
            color_video_path (str): Path to the original color video, used if not in grayscale mode.
            filter_type (str): Type of filter to use for tracking ("kalman" or "particle").
            show_particles (bool): Boolean representing if the particle positions are shown in the output.
        """
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

    evaluator = None
    if ground_truth_path is not None:
        evaluator = SyntheticEvaluator(ground_truth_path)


    tracking_on_video(new_cap, read_cap, out, grayscale, show_particles, filter_type, evaluator)

    new_cap.release()
    read_cap.release()
    out.release()

    cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    """
        Parses arguments for the script.

        Arguments:
            filter_type (str): The type of filter to use for tracking ("kalman" or "particle").
            video_path (str): Path to the video file.

        Optional Arguments:
            --original_video_path (str): Path to the original color video for non-grayscale processing.
            --output_video_path (str): Path to store the new created video file.
            --show_particles (bool): A boolean to signal if the video will show particles.

        Returns:
            argparse.Namespace: Parsed arguments with their values.
    """

    parser = argparse.ArgumentParser(description='Process video and evaluate results.')

    parser.add_argument("filter_type", type=str, help='Specify filter_type: kalman/particle')
    parser.add_argument('video_path', type=str, help='Path to the video file')

    parser.add_argument('--original_video_path', type=str, help='Path to the original video file')
    parser.add_argument('--output_video_path', type=str,
                        help='Path to store the new created video file, if empty it will be stored in the current '
                             'directory')
    parser.add_argument('--evaluation_file_path', type=str, help='Path to the ground truth file')
    parser.add_argument('--show_particles', action='store_true',
                        help=' A boolean to signal if the video will show particles')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.output_video_path is None:
        args.output_video_path = "./data/videos/output_tracked_video.mp4"

    if args.show_particles is None:
        args.show_particles = False

    use_filter_on_video(args.video_path, args.output_video_path, args.original_video_path, args.filter_type, args.evaluation_file_path ,
                        args.show_particles)
