import torch.optim as optim
import os
import clip
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import cv2
from torchvision.transforms import ToTensor
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,  ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import FastaiLRFinder


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Used device: ', device)

#Load CLIP model - ViT B16 (need to pip install clip first)
model, preprocess = clip.load('ViT-B/16', device, jit=False)

#Load the dataset
class ImageTitleDataset(Dataset):
    def __init__(self, list_video_path, list_labels, class_names, transform_image):
        #To handle the parent class
        super().__init__()
        #Initalize image paths and corresponding texts
        self.video_path = list_video_path
        #Initialize labels (0 or 1)
        self.labels = list_labels
        #Initialize class names (no smoke or smoke in prompts)
        self.class_names = class_names
        #Transform the image to the required tensor
        self.transform_image = transform_image

    @staticmethod
    #Function to create a square-shaped image from the video (similar to 1 long image)
    def preprocess_video_to_image_grid_version(video_path, num_rows=6, num_cols=6):
        #Open the video file
        video = cv2.VideoCapture(video_path)
        #Create list for extracted frames
        frames = []
        #Handle if video can't be opened
        if not video.isOpened():
            print("Error: Could not open video file")
        else:
            while True:
                is_read, frame = video.read()
                if not is_read:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            video.release()
        
        #If the video has a different amount of frames from 36, then produce a long image
        if len(frames) != 36:
            print("Num of frames are not 36")
            print("Num of frames for video on ", video_path, "is ", len(frames))
            concatenated_frames = np.concatenate(frames, axis=1)
        else:
            #Create  and store rows in the grids
            rows_list = []
            for i in range(num_rows):
                #create rows from the frames using indexes -- for example, if i=0, then between the 0th and 6th frame
                row = np.concatenate(frames[i * num_cols: (i + 1) * num_cols], axis=1)
                rows_list.append(row)
            
            #Concatenate grid vertically to create a single square-shaped image from the smoke video
            concatenated_frames = np.concatenate(rows_list, axis=0)
        return concatenated_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #Tranform videos into images and preprocess with clip's function
        video_path = self.video_path[idx]
        image = self.preprocess_video_to_image_grid_version(video_path)
        image = Image.fromarray(image)
        image = self.transform_image(image)
        #Get the corresponding class names and tokenize
        true_label = self.labels[idx]
        label = self.class_names[true_label]
        label = clip.tokenize(label, context_length=77, truncate=True)
        return image, label, true_label


#Define training, validation and test data
def load_data(split_path):
    with open(split_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_data = load_data('data/split/metadata_train_split_by_date.json')


#Prepare the list of video file paths and labels
def prepare_paths_labels(data, base_path):
    list_video_path = [os.path.join(base_path, f"{fn}.mp4") for fn in data['file_name']]
    list_labels = [int(label) for label in data['label']]
    return list_video_path, list_labels

base_path = "/../projects/0/prjs0930/data/merged_videos/"
train_list_video_path, train_list_labels = prepare_paths_labels(train_data, base_path)


#Define class names in a list - it needs prompt engineering
#class_names = ["a photo of a factory with no smoke", "a photo of a smoking factory"] #1
#class_names = ["a series picture of a factory with a shut down chimney", "a series picture of a smoking factory chimney"] #- 2
#class_names = ["a photo of factories with clear sky above chimney", "a photo of factories emitting smoke from chimney"] #- 3
#class_names = ["a photo of a factory with no smoke", "a photo of a factory with smoke emission"] #- 4
class_names = ["a series picture of a factory with clear sky above chimney", "a series picture of a smoking factory"] #- 5
#class_names = ["a series picture of a factory with no smoke", "a series picture of a smoking factory"] #- 6
#class_names = ["a sequental photo of an industrial plant with clear sky above chimney, created from a video", "a sequental photo of an industrial plant emiting smoke from chimney, created from a video"]# - 7
#class_names = ["a photo of a shut down chimney", "a photo of smoke chimney"] #-8
#class_names = ["The industrial plant appears to be in a dormant state, with no smoke or emissions coming from its chimney. The air around the facility is clear and clean.","The smokestack of the factory is emitting dark or gray smoke against the sky. The emissions may be a result of industrial activities within the facility."] #-9
#class_names = ["a photo of an industrial site with no visible signs of pollution", "a photo of a smokestack emitting smoke against the sky"] #-10
#class_names = ['no smoke', 'smoke'] #-11
#class_names = ['a photo of an industrial facility, emitting no smoke', 'a photo of an industrial facility, emitting smoke'] #12

# Define input resolution
input_resolution = (224, 224)

# Define the transformation pipeline - from CLIP preprocessor without random crop augmentation
transform_steps = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

# Create dataset and data loader for training, validation and testing
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, class_names, transform_steps)

print('Datasets created')

#Create dataloader fot training, validation and testig

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

print('Dataloaders created')


optimizer = optim.Adam(model.parameters(), lr=1e-8)
criterion = nn.CrossEntropyLoss()

def train_step(engine, batch):
    model.train()
    inputs, labels, true_label = batch
    inputs = inputs.to(device)
    optimizer.zero_grad()
    text_inputs = clip.tokenize(class_names).to(device)
    text_inputs = text_inputs.squeeze(dim = 1)
    outputs, _ = model(inputs, text_inputs)
    ground_truth = torch.tensor(true_label, dtype=torch.long, device=device)
    loss = criterion(outputs, ground_truth)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(train_step)

lr_finder = FastaiLRFinder()
to_save = {"model": model, "optimizer": optimizer}

with lr_finder.attach(trainer, to_save=to_save, start_lr=1e-8, end_lr=10.0) as trainer_with_lr_finder:
    trainer_with_lr_finder.run(train_dataloader)

# Get lr_finder results
results = lr_finder.get_results()

# Plot lr_finder results
lr_finder.plot()

# Get lr_finder suggestion for lr
suggested_lr = lr_finder.lr_suggestion()
print(f"Suggested learning rate: {suggested_lr}")


ax = lr_finder.plot(skip_end=0)
ax.figure.savefig("output.jpg")
