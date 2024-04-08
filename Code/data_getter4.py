import cv2
import os 

# Global variables to store coordinates
coordinates = []

def mouse_callback(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))

def draw_rectangle(image, coordinates):
    if len(coordinates) == 2:
        # Draw rectangle using the two coordinates
        cv2.rectangle(image, coordinates[0], coordinates[1], (0, 255, 0), 2)

def crop_and_save(image, coordinates, output_file):
    if len(coordinates) == 2:
        # Crop the region defined by the coordinates
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        cropped_image = image[y1:y2, x1:x2]

        # Save the cropped image to file
        cv2.imwrite(output_file, cropped_image)
        print("Cropped image saved successfully.")

def visualize_video(video_path, output_file):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    paused = False
    while True:
        if not paused:
            # Read a frame from the video
            ret, frame = cap.read()

            # Check if frame is successfully read
            if not ret:
                break

            # Draw the rectangle if both coordinates are available
            draw_rectangle(frame, coordinates)

            # Display the frame
            cv2.imshow('Video', frame)

        # Check for key press
        key = cv2.waitKey(30)

        # Pause/Unpause the video when 'p' is pressed
        if key == ord('p'):
            paused = not paused

            # If unpaused, crop and save the region
            if not paused:
                crop_and_save(frame, coordinates, output_file)
                coordinates.clear()  # Clear coordinates after saving

        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_file = "cropped_image.jpg"  # Change this to the desired output file path

    videos_path = r"C:\Users\PC KAIJU\ConceivableProjectsTools\FrameSampler\coc_follicular_h\ARDUcam\v10"
    video_name = "CP001-P032-1B~C-P2_dish_3-pipette-1.mp4"
    video = os.path.join(videos_path, video_name)

    output_path = os.path.join(videos_path, video_name.split(".")[0], output_file)

    # Set up mouse callback to capture coordinates
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)

    visualize_video(video, output_path)