#Import packages
import os
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from datetime import datetime

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda") # use CUDA device
#elif torch.backends.mps.is_available():
#    device = torch.device("mps") # use MacOS GPU device (e.g., for M2 chips)
else:
    device = torch.device("cpu") # use CPU device
print('Used device: ', device)

import torch
import torch.nn as nn
from torchvision import models

class MobileNetV3Small_RNN(nn.Module):
    def __init__(self, num_classes, rnn_type="LSTM"):
        super(MobileNetV3Small_RNN, self).__init__()
        #load the model
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)

        #extract features from final layer - pooling is exluded
        self.feature_extractor = self.mobilenet.features

        #Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        #get the number of features output by MobileNetV3 - for input for RNN
        self.num_features = self.mobilenet.classifier[0].in_features

        #room for rnn type choice: LSTM or GRU
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(self.num_features, hidden_size=256, num_layers=1, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(self.num_features, hidden_size=256, num_layers=1, batch_first=True)
        else:
            raise ValueError("Invalid RNN type. Choose 'LSTM' or 'GRU'.")

        #final classification layer - to get logits for the two classes
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()

        #accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. 
        #reshape input for feature extraction - mobilenet can only take images (4 d)
        c_in = x.view(batch_size * timesteps, C, H, W)
        
        #extract features with mobilenet
        features = self.feature_extractor(c_in)
        
        #pooling - using the same one as in the mobilenet architecture
        #lstm layer needs a 3D tensor, with shape (batch, timesteps, feature)
        features = self.pool(features).view(batch_size, timesteps, -1)

        #get rnn output by passing the features to the selected rnn
        rnn_out, _ = self.rnn(features)
        
        #batch, timesteps, output features
        #only select the last of the timesteps as it holds the information of the whole video
        last_output = rnn_out[:, -1, :]
        logits = self.fc(last_output)
        
        return logits


model = MobileNetV3Small_RNN(num_classes=2, rnn_type="LSTM")
model = model.to(device)

# Load the dataset
class ImageTitleDataset(Dataset):
    def __init__(self, list_video_path, list_labels, transform_image):
        #to handle the parent class
        super().__init__()
        #Initalize image paths and corresponding texts
        self.video_path = list_video_path
        #Initialize labels (0 or 1)
        self.labels = list_labels
        #Transform images based on defined transformation
        self.transform_image = transform_image

    @staticmethod
    #Function to create a square-shaped image from the video (similar to 1 long image)

    def preprocess_video_to_a_set_of_images(video_path):
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
        
        return frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #tranform videos into images and preprocess with defined transform function
        video_path = self.video_path[idx]
        frames = self.preprocess_video_to_a_set_of_images(video_path)
        frames = [self.transform_image(Image.fromarray(frame)) for frame in frames]
        frames = torch.stack(frames)

        #get the corresponding class names
        label = self.labels[idx]
        return frames, label
    

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
input_resolution = (256, 256)

# Define the transformation pipeline - from CLIP preprocessor without random crop augmentation
train_transform = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and data loader for training, validation and testing
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, train_transform)
val_dataset = ImageTitleDataset(val_list_video_path, val_list_labels, val_test_transform)
test_dataset = ImageTitleDataset(test_list_video_path, test_list_labels, val_test_transform)

print('Datasets created')

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('Dataloaders created')

num_epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*num_epochs)


best_te_loss = 1e5
best_ep = -1
early_stopping_counter = 0
early_stopping_patience = 4
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
    step = 0
    tr_loss = 0
    model.train()
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    epoch_train_correct = 0
    epoch_train_total = 0
    for batch in pbar:
        step += 1

        # Extract images and labels from the batch
        images, labels = batch 

        # Move images and texts to the specified device (CPU or GPU)
        images= images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        batch_loss = loss(predictions, labels)
        tr_loss += batch_loss.item()

        # Calculate accuracy
        probabilities = torch.argmax(predictions, dim=1)

        correct = (probabilities == labels).sum().item()
        total = labels.size(0)
        epoch_train_correct += correct
        epoch_train_total += total

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        #scheduler.step()
        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {batch_loss.item():.4f}, Current Learning rate: {optimizer.param_groups[0]['lr']}")
    tr_loss /= step
    train_accuracy = epoch_train_correct / epoch_train_total
    print('Train accuracy: ', train_accuracy)
    train_accuracies.append(train_accuracy)
    train_losses.append(tr_loss)

    print('Validation loop starts')
    model.eval()
    step = 0
    te_loss = 0
    all_preds = []
    all_labels = []
    epoch_val_correct = 0
    epoch_val_total = 0
    with torch.no_grad():
        vbar = tqdm(val_dataloader, total=len(val_dataloader))
        i = 0
        for batch in vbar:
            step += 1
            images, labels = batch
            images= images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            val_loss = loss(predictions, labels)
            te_loss += val_loss.item()

            pred_labels = predictions.argmax(dim=1)

            correct = (pred_labels == labels).sum().item()
            total = labels.size(0)
            epoch_val_correct += correct
            epoch_val_total += total

            # Append predicted labels and ground truth labels
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Append loss
            val_losses.append(val_loss.item())
        
            # Update the progress bar with the current epoch and loss
            vbar.set_description(f"Validation: {i}/{len(val_dataloader)}, Validation loss: {val_loss.item():.4f}")
            i+=1

        te_loss /= step
        val_accuracy = epoch_val_correct / epoch_val_total
        print("Validation accuracy: ", val_accuracy)
        val_losses.append(te_loss)
        val_accuracies.append(val_accuracy)
        train_losses.append(te_loss)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Print or visualize the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    #get evaluation metrics:

    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f_score= f1_score(all_labels, all_preds, average='binary')
    acc = accuracy_score(all_labels, all_preds)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f_score:.4f}")

    if te_loss < best_te_loss:
        best_te_loss = te_loss
        best_ep = epoch
        torch.save(model.state_dict(), "../light_cnn_best_model.pt")
        early_stopping_counter = 0 
    else:
        early_stopping_counter += 1

    print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")

    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping after {epoch + 1} epochs.")
        break
    
print(f"best epoch {best_ep+1}, best te_loss {best_te_loss}")
torch.save(model.state_dict(), "../light_cnn_last_model.pt")

print("start testing")
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    start_time = datetime.now()
    tbar = tqdm(test_dataloader, total=len(test_dataloader))
    i = 0
    for batch in tbar:
        images, labels = batch
        images= images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        test_loss = loss(predictions, labels)

        pred_labels = predictions.argmax(dim=1)

        # Append predicted labels and ground truth labels
        test_preds.extend(pred_labels.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
    
        # Update the progress bar with the current epoch and loss
        tbar.set_description(f"Testing: {i}/{len(test_dataloader)}, Test loss: {test_loss.item():.4f}")
        i+=1
    end_time = datetime.now()
    print('Start time: ', start_time)
    print('Ending time: ', end_time)
    print('Overall time: ', end_time-start_time)

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print or visualize the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

#get evaluation metrics:

precision = precision_score(test_labels, test_preds, average='binary')
recall = recall_score(test_labels, test_preds, average='binary')
f_score= f1_score(test_labels, test_preds, average='binary')
acc = accuracy_score(test_labels, test_preds)

# Print or log the metrics
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f_score:.4f}")

print("CLIP model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")

# Classification report
target_names = ['class 0', 'class 1']
print(classification_report(test_labels, test_preds, target_names=target_names))

# Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_validation_accuracy.png')
plt.close()

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Loss')
plt.plot(val_accuracies, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss.png')
plt.close()