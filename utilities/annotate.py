import cv2
import json
from typing import Dict, List


def annotate_video(video_path: str, output_json: str, annotation_file: str) -> None:
    """
    Annotate a video by clicking on frames and save annotations to a JSON file.

    Parameters:
    video_path (str): Path to the video file.
    output_json (str): Path to save the output JSON file.
    """
    try:
        with open(annotation_file, "r") as file:
            annotations: Dict[int, List[Dict[str, int]]] = json.load(file)
            annotations = {int(frame_number): value for frame_number, value in annotations.items()}
        print(f"Loaded existing annotations from {annotation_file}")
    except FileNotFoundError:
        annotations = {}
        print(f"No existing annotation file found. Starting fresh.")

    current_id = 1
    current_frame = 214

    def click_event(event, x, y, flags, param):
        """
        Callback function to handle mouse clicks and save coordinates.
        """
        nonlocal current_id, current_frame, annotations
        if event == cv2.EVENT_LBUTTONDOWN:
            # Save the clicked point
            x = x // 2
            y = y // 2
            frame_annotations = annotations.setdefault(current_frame, [])
            frame_annotations.append({"id": current_id, "x": x, "y": y})
            print(f"Frame {current_frame}: Added annotation - ID: {current_id}, X: {x}, Y: {y}")

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", click_event)
    a = True
    while True:
        # Set the video to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("End of video or unable to read frame.")
            break

        # Display frame number and current ID on the frame
        display_text = f"Frame: {current_frame}, Current ID: {current_id}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the annotations for the current frame
        if current_frame in annotations:
            for annotation in annotations[current_frame]:
                x, y = annotation["x"], annotation["y"]
                cv2.circle(frame, (x*2, y*2), 5, (0, 0, 255), -1)  # Mark the point
                cv2.putText(frame, f'ID: {annotation["id"]}', (x*2 + 10, y*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)

        # Display the frame
        cv2.imshow("Video", frame)

        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF

        if key == ord('e'):  # Increase ID
            current_id += 1
            print(f"Current ID: {current_id}")

        elif key == ord('q'):  # Decrease ID
            if current_id > 1:
                current_id -= 1
            print(f"Current ID: {current_id}")

        elif key == 13:  # Enter key - Move to the next frame
            current_frame += 1
            print(f"Moved to frame {current_frame}")

        elif key == ord('b'):  # Go back to the previous frame
            if current_frame > 0:
                current_frame -= 1
            print(f"Moved to frame {current_frame}")

        elif key == ord('s'):  # Save and exit
            with open(output_json, "w") as json_file:
                json.dump(annotations, json_file, indent=4)
            print(f"Annotations saved to {output_json}")
            break

        elif key == ord('x'):  # Exit without saving
            print("Exiting without saving.")
            break

    with open(output_json, "w") as json_file:
        json.dump(annotations, json_file, indent=4)
    print(f"Annotations saved to {output_json}")
    cap.release()
    cv2.destroyAllWindows()


# Run the function
video_path = "C:\\SchoolApps\\Bakalarka\\Datasets\\vineyard_videos\\row_videos\\row_SG19_small.mp4"  # Replace with your video path
output_json = "annotations.json"      # Replace with your desired output path
annotation_file= "C:\\SchoolApps\\Bakalarka\\Bakalarka_kod\\data\\ground_truth_data\\SG_annotations.json"
annotate_video(video_path, output_json, annotation_file)