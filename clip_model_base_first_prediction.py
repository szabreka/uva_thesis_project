import torch
import clip
from PIL import Image
import cv2
import os

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
example_path = 'data/ijmond_videos/5PurGkmy0aw-1.mp4'

#define class names in a list - it need prompt engineering
class_names = ["a photo of a smoking factory", "a photo of a factory with no smoke above chimney"]

#Crete a list of images from video
def preprocess_video(video_path):
    # Open the video file
    # example video : video_path = 'data/ijmond_videos/5PurGkmy0aw-1.mp4'
    video = cv2.VideoCapture(video_path)
    frames = []
    if not video.isOpened():
        print("Error: Could not open video file")
    else:
        i = 1
        while True:
            ret, image = video.read()
            if ret == False:
                print('End of video reached during preprocessing')
                break
            frames.append(image)
            i += 1
        video.release()
    return frames


def vanilla_clip(video_path):
    #Create image list from video
    frames = preprocess_video(video_path)

    # Loop over each frame in video (36 frames in 1 video)
    i = 1
    prediction_list= []
    for frame in frames:
        # Read image and preprocess
        image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)

        # Prepare text inputs based on class names list
        text_inputs = clip.tokenize(class_names).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

        # Calculate similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        if class_names[indices] == "a photo of a factory with no smoke above chimney":
            prediction_list.append(0)
        else:
            prediction_list.append(1)       

        # Print predictions for each frame
        #print(f"\nPredictions for frame {i}:\n")
        #for value, index in zip(values, indices):
        #    print(f"{class_names[index]:>16s}: {100 * value.item():.2f}%")
        i+=1
    print(prediction_list)
    if sum(prediction_list) >3:
        print("The video contains smoke")

vanilla_clip(example_path)
