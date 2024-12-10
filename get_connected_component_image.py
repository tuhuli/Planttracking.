import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Threshold an image, calculate connected components, and filter them by size.')
    parser.add_argument('input_image', help='Path to the input image.')
    parser.add_argument('threshold', type=int, help='Threshold value (0-255).')
    parser.add_argument('output_image', help='Path to save the filtered image.')
    args = parser.parse_args()

    # Read the input image in grayscale
    image = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Error: Could not read the input image.')
        return

    # Apply threshold
    _, thresh = cv2.threshold(image, args.threshold, 255, cv2.THRESH_BINARY)

    # Compute connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Create an output image initialized to zeros (black)
    filtered_image = np.zeros_like(image)

    # Filter components by size
    for label in range(1, num_labels):  # Skip the background label (0)
        size = stats[label, cv2.CC_STAT_AREA]
        if 3000 <= size <= 10000:
            # Include the component in the output image
            filtered_image[labels == label] = 255

    # Save the filtered image
    cv2.imwrite(args.output_image, filtered_image)
    print(f'Filtered image saved as {args.output_image}')

if __name__ == '__main__':
    main()