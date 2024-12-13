The filter type and the path to the video with probability maps are needed to run the program.
The CNN output is located in data/videos. The video provided by the industry partner is 
in the original directory. The original video can also be found there and can be used as 
--output_video_path. The artificial videos are located in data/videos/synthetic.

```markdown
## Usage

To run the script: "python main.py 'kalman'/'particle' "data/videos/original/row_SG19_small_predikce.mp4" "
```

```markdown
## Arguments for extended use are:

--original_video_path "PATH_TO_ORIGINAL_VIDEO"   It will create a new video with detections.

--output_video_path "PATH_TO_NEW_VIDEO" The location with the new name of the video is set. The format for the new
video should be .mp4

--evaluation_file_path "PATH_TO_THE_FILE_WITH_GROUND_TRUTH_DATA"     is the file with the ground truth data for the
processed video. If not selected, the evaluation is not done.

--show_particles    Argument to signal if the video should show particles of particle filters.
```
