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

# Load the dataset
class ImageTitleDataset(Dataset):
    def __init__(self, list_video_path, list_labels, class_names, transform_image):
        #to handle the parent class
        super().__init__()
        #Initalize image paths and corresponding texts
        self.video_path = list_video_path
        #Initialize labels (0 or 1)
        self.labels = list_labels
        #Initialize class names (no smoke or smoke)
        self.class_names = class_names
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
        label = self.class_names[true_label]
        label = clip.tokenize(label, context_length=77, truncate=True)
        return image, label, true_label
    
#Define training, validation and test data
# Load the JSON metadata
with open('data/datasets/fourshot_dataset.json', 'r') as f:
    train_data = json.load(f)
with open('data/split/metadata_test_split_by_date.json', 'r') as f:
    test_data = json.load(f)


# Convert the datasets to a Pandas DataFrame
train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)


# Prepare the list of video file paths and labels
train_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in train_data['file_name']]
train_list_labels = [int(label) for label in train_data['label']]
test_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in test_data['file_name']]
test_list_labels = [int(label) for label in test_data['label']]

#Define class names in a list - it needs prompt engineering
#class_names = ["a photo of a factory with no smoke", "a photo of a smoking factory"] #1
#class_names = ["a series picture of a factory with a shut down chimney", "a series picture of a smoking factory chimney"] #- 2
#class_names = ["a photo of factories with clear sky above chimney", "a photo of factories emiting smoke from chimney"] #- 3
#class_names = ["a photo of a factory with no smoke", "a photo of a factory with smoke emission"] #- 4
class_names = ["a series picture of a factory with clear sky above chimney", "a series picture of a smoking factory"] #- 5
#class_names = ["a series picture of a factory with no smoke", "a series picture of a smoking factory"] #- 6
#class_names = ["a sequental photo of an industrial plant with clear sky above chimney, created from a video", "a sequental photo of an industrial plant emiting smoke from chimney, created from a video"]# - 7
#class_names = ["a photo of a shut down chimney", "a photo of smoke chimney"] #-8
#class_names = ["The industrial plant appears to be in a dormant state, with no smoke or emissions coming from its chimney. The air around the facility is clear and clean.","The smokestack of the factory is emitting dark or gray smoke against the sky. The emissions may be a result of industrial activities within the facility."] #-9
#class_names = ["a photo of an industrial site with no visible signs of pollution", "a photo of a smokestack emitting smoke against the sky"] #-10
#class_names = ['no smoke', 'smoke']

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
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, class_names, train_transform)
test_dataset = ImageTitleDataset(test_list_video_path, test_list_labels, class_names, test_transform)

print('Datasets created')

#Create dataloader fot training, validation and testig

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print('Dataloaders created')

# Function to convert model's parameters to FP32 format
#This is done so that our model loads in the provided memory
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

# Check if the device is set to CPU
if device == "cpu":
  model.float()

#Define number of epochs
num_epochs = 1

# Prepare the optimizer - the lr, betas, eps and weight decay are from the CLIP paper
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*num_epochs)

# Specify the loss functions - for images and for texts
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Model training
print('starts training')
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        # Extract images and texts from the batch
        images, labels, true_label = batch 

        # Move images and texts to the specified device (CPU or GPU)
        images= images.to(device)
        texts = labels.to(device)
        true_label = true_label.to(device)
        text_inputs = clip.tokenize(class_names).to(device)

        #Squeeze texts tensor to match the required size
        texts = texts.squeeze(dim = 1)
        text_inputs = text_inputs.squeeze(dim = 1)

        # Forward pass - Run the model on the input data (images and texts)
        logits_per_image, logits_per_text = model(images, text_inputs)

        #Transform logits to float to match required dtype 
        logits_per_image = logits_per_image.float()
        logits_per_text = logits_per_text.float()

        #Ground truth
        ground_truth = torch.tensor(true_label, dtype=torch.long, device=device)

        #Compute loss - contrastive loss to pull similar pairs closer together
        #total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text.T,ground_truth))/2

        #One image should match 1 label, but 1 label can match will multiple images (when single label classification)
        total_loss = loss_img(logits_per_image, ground_truth) 

        # Get and convert similarity scores to predicted labels
        similarity = logits_per_image.softmax(dim=-1)
        value, index = similarity.topk(1)
        
        #Convert values to numpy
        predicted_label = index.cpu().numpy()
        ground_truth_label = ground_truth.cpu().numpy()
        
        train_accuracy = accuracy_score(ground_truth_label, predicted_label)
        print('Train accuracy: ', train_accuracy)

        # Zero out gradients for the optimizer (Adam) - to prevent adding gradients to previous ones
        optimizer.zero_grad()

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
            #scheduler.step()
        else : 
            # Convert model's parameters to FP32 format, update, and convert back
            convert_models_to_fp32(model)
            optimizer.step()
            #scheduler.step()
            clip.model.convert_weights(model)
        # Update the progress bar with the current epoch and loss
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}, Current Learning rate: {optimizer.param_groups[0]['lr']}")

print('Start testing...')


model.eval()
test_losses = []
test_accs = []
test_preds = []
test_labels = []
with torch.no_grad():
        tbar = tqdm(test_dataloader, total=len(test_dataloader))
        i = 0
        for batch in tbar:
                # Extract images and texts from the batch
                images, labels, true_label = batch 
                # Move images and texts to the specified device
                images = images.to(device)
                texts = labels.to(device)
                true_label = true_label.to(device)
                text_inputs = clip.tokenize(class_names).to(device)
                texts = texts.squeeze(dim=1)
                text_inputs.squeeze(dim=1)

                # Forward pass
                logits_per_image, logits_per_text = model(images, text_inputs)

                #Transform logits to float to match required dtype 
                logits_per_image = logits_per_image.float()
                logits_per_text = logits_per_text.float()

                # Get and convert similarity scores to predicted labels - values are the probabilities, indicies are the classes
                similarity = logits_per_image.softmax(dim=-1)
                value, index = similarity.topk(1)

                ground_truth = torch.tensor(true_label, dtype=torch.long, device=device)

                #One image should match 1 label, but 1 label can match will multiple images (when single label classification)
                total_loss = loss_img(logits_per_image,ground_truth) 

                # Convert similarity scores to predicted labels
                predicted_label = index.cpu().numpy()
                ground_truth_label = ground_truth.cpu().numpy()

                # Append predicted labels and ground truth labels
                test_preds.append(predicted_label)
                test_labels.append(ground_truth_label)

                # Append loss
                test_losses.append(total_loss.item())

                # Update the progress bar with the current epoch and loss
                tbar.set_description(f"Testing: {i}/{len(test_dataloader)}, Test loss: {total_loss.item():.4f}")
                i+=1
    
# Convert lists of arrays to numpy arrays
all_labels_array = np.concatenate(test_labels)
all_preds_array = np.concatenate(test_preds)

# Convert to 1D arrays
all_labels_flat = all_labels_array.flatten()
all_preds_flat = all_preds_array.flatten()

# Ensure they are integers
all_labels_int = all_labels_flat.astype(int)
all_preds_int = all_preds_flat.astype(int)

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels_int, all_preds_int)

# Print or visualize the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

#get evaluation metrics:

precision = precision_score(all_labels_int, all_preds_int, average='binary')
recall = recall_score(all_labels_int, all_preds_int, average='binary')
f_score= f1_score(all_labels_int, all_preds_int, average='binary')
acc = accuracy_score(all_labels_int, all_preds_int)

# Print or log the metrics
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f_score:.4f}")

def get_features(dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, class_names, labels  in dataloader:
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

#test_features, test_labels = get_features(test_dataloader)
print('Test features created')

def visualize_features(features, labels, title):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.title(title)
    plt.savefig('clip_fourshot_features.png')
    plt.close()

#visualize_features(test_features, test_labels, 'Test Features')