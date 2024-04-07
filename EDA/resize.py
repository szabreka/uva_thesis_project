import cv2
import os

#define paths
file = "HgNO44L2INY-1"
input_path = f'/Users/szaboreka/Documents/UvA/Thesis/uva_thesis_project/data/ijmond_videos/{file}.mp4'
output_dictionary = '/Users/szaboreka/Documents/UvA/Thesis/uva_thesis_project/data/ijmond_videos_180_180'

#read file
cap = cv2.VideoCapture(input_path)

#define parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
dim = (180, 180)

#create output dictionary if not exists
if not os.path.exists(output_dictionary):
    os.makedirs(output_dictionary)
out = cv2.VideoWriter(f'{output_dictionary}/{file}.mp4', fourcc, fps, dim)

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

print(f"Video successfully resized and saved")

vid1 = cv2.VideoCapture(f'{output_dictionary}/{file}.mp4')

# Get the dimensions of the first example video
print("\nExample video:")
ret, frame = vid1.read()
shape = frame.shape
fps = int(vid1.get(cv2.CAP_PROP_FPS))
total_frames = int(vid1.get(cv2.CAP_PROP_FRAME_COUNT))
video_length_seconds = total_frames / fps
print(f"Shape: {shape}")
print(f"Frame rate: {fps} frames per second")
print(f"Total number of frames: {total_frames}")
print(f"Total length in second: {video_length_seconds}")