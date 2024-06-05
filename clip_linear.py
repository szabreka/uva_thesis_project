#Import packages
import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import cv2
from torchvision.transforms import ToTensor
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda") # use CUDA device
#elif torch.backends.mps.is_available():
#    device = torch.device("mps") # use MacOS GPU device (e.g., for M2 chips)
else:
    device = torch.device("cpu") # use CPU device
print('Used device: ', device)

#Load CLIP model - ViT B32
model, preprocess = clip.load('ViT-B/16', device, jit=False)

state_dict = torch.load('../clip_fully_supervised/fs_best_model_30e_5p.pt', map_location=device)
model.load_state_dict(state_dict)

# Load the dataset
class ImageTitleDataset(Dataset):
    def __init__(self, list_video_path, list_labels, transform_image):
        #to handle the parent class
        super().__init__()
        #Initalize image paths and corresponding texts
        self.video_path = list_video_path
        #Initialize labels (0 or 1)
        self.labels = list_labels
        #Transform to tensor
        #self.transforms = ToTensor()
        self.transform_image = transform_image

    @staticmethod
    #Function to create a square-shaped image from the video (similar to 1 long image)
    #To do: what if the video has more frames than 36?
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
        
        if len(frames) != 36:
            print("Num of frames are not 36")
            print("Num of frames for video on ", video_path, "is ", len(frames))
        
        # Create  and store rows in the grids
        rows_list = []
        for i in range(num_rows):
            #create rows from the frames using indexes -- for example, if i=0, then between the 0th and 6th frame
            row = np.concatenate(frames[i * num_cols: (i + 1) * num_cols], axis=1)
            rows_list.append(row)
        
        # Concatenate grid vertically to create a single square-shaped image from the smoke video
        concatenated_frames = np.concatenate(rows_list, axis=0)
        return concatenated_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #tranform videos into images and preprocess with clip's function
        video_path = self.video_path[idx]
        image = self.preprocess_video_to_image_grid_version(video_path)
        image = Image.fromarray(image)
        image = self.transform_image(image)
        #image = preprocess(image)
        #get the corresponding class names and tokenize
        true_label = self.labels[idx]
        #label = self.class_names[true_label]
        #label = clip.tokenize(label, context_length=77, truncate=True)
        return image, true_label
    
#Define training, validation and test data
# Load the JSON metadata
with open('data/split/metadata_train_split_by_date.json', 'r') as f:
    train_data = json.load(f)
with open('data/split/metadata_validation_split_by_date.json', 'r') as f:
    val_data = json.load(f)
with open('data/split/metadata_test_split_by_date.json', 'r') as f:
    test_data = json.load(f)


# Convert the datasets to a Pandas DataFrame
train_data = pd.DataFrame(train_data)
val_data = pd.DataFrame(val_data)
test_data = pd.DataFrame(test_data)


# Prepare the list of video file paths and labels
train_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in train_data['file_name']]
train_list_labels = [int(label) for label in train_data['label']]
val_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in val_data['file_name']]
val_list_labels = [int(label) for label in val_data['label']]
test_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in test_data['file_name']]
test_list_labels = [int(label) for label in test_data['label']]

# Define input resolution
input_resolution = (224, 224)

# Define the transformation pipeline - from CLIP preprocessor without random crop augmentation
train_transform = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

val_transform = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

test_transform = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

# Create dataset and data loader for training, validation and testing
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, train_transform)
val_dataset = ImageTitleDataset(val_list_video_path, val_list_labels, val_transform)
test_dataset = ImageTitleDataset(test_list_video_path, test_list_labels, test_transform)

print('Datasets created')

#Create dataloader fot training, validation and testig

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print('Dataloaders created')

def get_features(dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels  in dataloader:
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train_dataloader)
print('Train feaures created')
val_features, val_labels = get_features(val_dataloader)
print('Validation features created')
test_features, test_labels = get_features(test_dataloader)
print('Test features created')

def visualize_features(features, labels, title):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.title(title)
    plt.savefig('clip_features_linear_30e_5p_best_model.png')
    plt.close()

visualize_features(test_features, test_labels, 'Test Features')

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)

# Optionally, use early stopping based on the validation set performance
# For example, if validation accuracy does not improve for several epochs, stop training


# Training
print('Training')
classifier.fit(train_features, train_labels)

train_predictions = classifier.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)
train_precision = precision_score(train_labels, train_predictions)
train_recall = recall_score(train_labels, train_predictions)
train_f1 = f1_score(train_labels, train_predictions)

print(f"Train Accuracy = {train_accuracy:.3f}")
print(f"Train Precision = {train_precision:.3f}")
print(f"Train Recall = {train_recall:.3f}")
print(f"Train F1 Score = {train_f1:.3f}")

# Validation
print('Validation')
val_predictions = classifier.predict(val_features)

val_accuracy = accuracy_score(val_labels, val_predictions)
val_precision = precision_score(val_labels, val_predictions)
val_recall = recall_score(val_labels, val_predictions)
val_f1 = f1_score(val_labels, val_predictions)

print(f"Validation Accuracy = {val_accuracy:.3f}")
print(f"Validation Precision = {val_precision:.3f}")
print(f"Validation Recall = {val_recall:.3f}")
print(f"Validation F1 Score = {val_f1:.3f}")

    


start_time = datetime.now()
# Evaluate the trained classifier on the test set
test_predictions = classifier.predict(test_features)

test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions)
test_recall = recall_score(test_labels, test_predictions)
test_f1 = f1_score(test_labels, test_predictions)

end_time = datetime.now()
print('Start time: ', start_time)
print('Ending time: ', end_time)
print('Overall time: ', end_time-start_time)

print(f"Test Accuracy = {test_accuracy:.3f}")
print(f"Test Precision = {test_precision:.3f}")
print(f"Test Recall = {test_recall:.3f}")
print(f"Test F1 Score = {test_f1:.3f}")

conf_matrix = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

print("CLIP model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")