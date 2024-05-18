import cv2
import os

# Directory containing the .npy files
data_dir = "/../../projects/0/prjs0930/data/ijmond_videos/"

# List all files in the director
files = os.listdir(data_dir)

#define paths
#file = "HgNO44L2INY-1"
#input_path = f'/Users/szaboreka/Documents/UvA/Thesis/uva_thesis_project/data/ijmond_videos/{file}.mp4'
output_dictionary = '/../../projects/0/prjs0930/data/ijmond_videos_resized/'

#create output dictionary if not exists
if not os.path.exists(output_dictionary):
    os.makedirs(output_dictionary)


for file in files:

    input_path = os.path.join(data_dir, file)

    #read file
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        continue

    #define parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    dim = (180, 180)

    # Define full path for output video
    output_path = os.path.join(output_dictionary, file)

    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, dim)

    #resize and save video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # Write the resized frame to the output video
        out.write(resized)

    # Release the video capture and writer and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video successfully resized and saved: {output_path}")

