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
import torch.optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                frames.append(frame)
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

#Define class names in a list - it needs prompt engineering
#class_names = ["a photo of a factory with no smoke", "a photo of a smoking factory"] #1
#class_names = ["a series picture of a factory with a shut down chimney", "a series picture of a smoking factory chimney"] #- 2
#class_names = ["a photo of factories with clear sky above chimney", "a photo of factories emiting smoke from chimney"] #- 3
#class_names = ["a photo of a factory with no smoke", "a photo of a smoking factory"] #- 4
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
transform = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

# Create dataset and data loader for training, validation and testing
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, class_names, transform)
val_dataset = ImageTitleDataset(val_list_video_path, val_list_labels, class_names, transform)
test_dataset = ImageTitleDataset(test_list_video_path, test_list_labels, class_names, transform)

print('Datasets created')

#Create dataloader fot training and validation

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

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

# Prepare the optimizer - the lr, betas, eps and weight decay are from the CLIP paper
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

# Specify the loss functions - for images and for texts
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Model training
num_epochs = 50
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
    similarity = logits_per_image[0].softmax(dim=-1)
    value, index = similarity[0].topk(1)
    
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
    else : 
        # Convert model's parameters to FP32 format, update, and convert back
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)
    # Update the progress bar with the current epoch and loss
    pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

  model.eval()
  val_losses = []
  val_accs = []
  all_preds = []
  all_labels = []
  with torch.no_grad():
      for batch in tqdm(val_dataloader, total=len(val_dataloader)):
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
            all_preds.append(predicted_label)
            all_labels.append(ground_truth_label)

            # Append loss
            val_losses.append(total_loss.item())

            val_accuracy = accuracy_score(ground_truth_label, predicted_label)
            print('Validation accuracy: ', val_accuracy)

            val_precision = precision_score(ground_truth_label, predicted_label, average='binary')
            print('Validation_precision: ', val_precision)
  
  # Calculate confusion matrix
  conf_matrix = confusion_matrix(all_labels, all_preds)

  # Print or visualize the confusion matrix
  print("Confusion Matrix:")
  print(conf_matrix)

print('Start testing')
def test_clip(dataset):
    predicted_labels= []
    ground_truths = []
    # Loop over each image in dataloader
    for rows in dataset:
        
        images, labels, true_label = rows

        # Move images and texts to the specified device
        images = images.unsqueeze(0).to(device)
        texts = labels.to(device)
        #true_label = true_label.to(device)
        text_inputs = clip.tokenize(class_names).to(device)
        texts = texts.squeeze(dim=1)
        text_inputs.squeeze(dim=1)

        # Calculate features
        #with torch.no_grad():
        #    image_features = model.encode_image(images)
        #    text_features = model.encode_text(text_inputs)

        # Calculate similarity
        #image_features /= image_features.norm(dim=-1, keepdim=True)
        #text_features /= text_features.norm(dim=-1, keepdim=True)
        #similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        #print(similarity)

        #It's the same as the similarity:
        logits_per_image, logits_per_text = model(images, text_inputs)

        similarity = logits_per_image.softmax(dim=-1)
        value, index = similarity.topk(1)

        ground_truth = torch.tensor(true_label, dtype=torch.long, device=device)

        # Convert similarity scores to predicted labels
        predicted_label = index.cpu().numpy()
        ground_truth_label = ground_truth.cpu().numpy()

        predicted_labels.append(predicted_label)
        ground_truths.append(ground_truth_label)
    
    # Compute accuracy
    accuracy = accuracy_score(ground_truths, predicted_labels)

    # Compute precision
    precision = precision_score(ground_truths, predicted_labels, average='binary')

    # Compute recall
    recall = recall_score(ground_truths, predicted_labels, average='binary')

    # Compute F1 score
    f1 = f1_score(ground_truths, predicted_labels, average='binary')
    
    # Print or log the metrics
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

test_clip(test_dataset)
