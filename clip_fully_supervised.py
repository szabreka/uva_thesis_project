#Import packages
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

#Define the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Used device: ', device)

#Load CLIP model (need to pip install clip first)
model, preprocess = clip.load('ViT-B/16', device, jit=False)


#Load the dataset
class ImageTitleDataset(Dataset):
    def __init__(self, list_video_path, list_labels, transform_image):
        #To handle the parent class
        super().__init__()
        #Initalize image paths and corresponding texts
        self.video_path = list_video_path
        #Initialize labels (0 or 1)
        self.labels = list_labels
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
    
        return image, true_label


#Define training, validation and test data
def load_data(split_path):
    with open(split_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

train_data = load_data('data/split/metadata_train_split_by_date.json')
val_data = load_data('data/split/metadata_validation_split_by_date.json')
test_data = load_data('data/split/metadata_test_split_by_date.json')

'''train_data = load_data('data/split/metadata_train_split_3_by_camera.json')
val_data = load_data('data/split/metadata_validation_split_3_by_camera.json')
test_data = load_data('data/split/metadata_test_split_3_by_camera.json')'''

#Prepare the list of video file paths and labels
def prepare_paths_labels(data, base_path):
    list_video_path = [os.path.join(base_path, f"{fn}.mp4") for fn in data['file_name']]
    list_labels = [int(label) for label in data['label']]
    return list_video_path, list_labels

base_path = "/../projects/0/prjs0930/data/merged_videos/"
train_list_video_path, train_list_labels = prepare_paths_labels(train_data, base_path)
val_list_video_path, val_list_labels = prepare_paths_labels(val_data, base_path)
test_list_video_path, test_list_labels = prepare_paths_labels(test_data, base_path)


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
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, transform_steps)
val_dataset = ImageTitleDataset(val_list_video_path, val_list_labels, transform_steps)
test_dataset = ImageTitleDataset(test_list_video_path, test_list_labels, transform_steps)

print('Datasets created')

#Create dataloader fot training, validation and testig

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('Dataloaders created')

#Function to convert model's parameters to FP32 format - this is done so that our model loads in the provided memory
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

# Check if the device is set to CPU
if device == "cpu":
  model.float()

#Define number of epochs
num_epochs = 5

# Prepare the optimizer - the lr, betas, eps and weight decay are from the CLIP paper
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
#Different tried schedulers:
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*num_epochs)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)


# Specify the loss function - for images only
loss_img = nn.CrossEntropyLoss()

best_te_loss = 1e5
best_ep = -1
early_stopping_counter = 0
early_stopping_patience = 5
early_stopping_threshold = 0.03
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

#Get image features 
def get_features(dataloader):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels  in dataloader:
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


#Visualize features using PCA
def visualize_features(features, labels, title, epoch):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'clip_fullysupervised_features_{epoch}.png')
    plt.close()

#Model training
print('starts training')
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

        #Extact images, class_names and labels from batch
        images, true_label = batch 

        #Move images and texts to the specified device (CPU or GPU)
        images= images.to(device)
        true_label = true_label.to(device)
        text_inputs = clip.tokenize(class_names).to(device)

        #Squeeze texts tensor to match the required size
        text_inputs = text_inputs.squeeze(dim = 1)

        #Forward pass - Run the model on the input data (images and texts from class names)
        logits_per_image, logits_per_text = model(images, text_inputs)

        #Transform logits to float to match required dtype 
        logits_per_image = logits_per_image.float()
        logits_per_text = logits_per_text.float()

        #Ground truth
        ground_truth = torch.tensor(true_label, dtype=torch.long, device=device)

        #One image should match 1 label, but 1 label can match will multiple images (when single label classification)
        total_loss = loss_img(logits_per_image, ground_truth) 

        # Get and convert similarity scores to predicted labels
        similarity = logits_per_image.softmax(dim=-1)
        value, index = similarity.topk(1)
        
        #Convert values to numpy
        predicted_label = index.squeeze().cpu().numpy()
        ground_truth_label = ground_truth.cpu().numpy()

        #get correct and total predictions for training accuracy 
        correct = (predicted_label == ground_truth_label).sum().item()
        total = len(true_label)
        epoch_train_correct += correct
        epoch_train_total += total

        #Zero out gradients for the optimizer - to prevent adding gradients to previous ones
        optimizer.zero_grad()

        # Backward pass
        total_loss.backward()
        tr_loss += total_loss.item()
        if device == "cpu":
            optimizer.step()
        else : 
            #Convert model's parameters to FP32 format, update, and convert back
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
        #Update the progress bar with the current epoch and loss
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, Current Learning rate: {optimizer.param_groups[0]['lr']}")
    tr_loss /= step
    train_accuracy = epoch_train_correct / epoch_train_total
    print('Training accuracy: ', train_accuracy)
    train_losses.append(tr_loss)
    train_accuracies.append(train_accuracy)

    print('Validation loop starts...')
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
                #Extract images, class names and labels from the batch
                images, true_label = batch 
                #Move images and texts to the specified device
                images = images.to(device)
                true_label = true_label.to(device)
                text_inputs = clip.tokenize(class_names).to(device)
                text_inputs.squeeze(dim=1)

                # Forward pass
                logits_per_image, logits_per_text = model(images, text_inputs)

                #Transform logits to float to match required dtype 
                logits_per_image = logits_per_image.float()
                logits_per_text = logits_per_text.float()

                #Get and convert similarity scores to predicted labels - values are the probabilities, indicies are the classes
                similarity = logits_per_image.softmax(dim=-1)
                value, index = similarity.topk(1)

                ground_truth = torch.tensor(true_label, dtype=torch.long, device=device)

                #One image should match 1 label, but 1 label can match will multiple images (when single label classification)
                total_loss = loss_img(logits_per_image,ground_truth) 
                te_loss += total_loss.item()

                #Convert similarity scores to predicted labels
                predicted_label = index.squeeze().cpu().numpy()
                ground_truth_label = ground_truth.cpu().numpy()

                correct = (predicted_label == ground_truth_label).sum().item()
                total = len(true_label)
                epoch_val_correct += correct
                epoch_val_total += total

                # Append predicted labels and ground truth labels
                all_preds.append(predicted_label)
                all_labels.append(ground_truth_label)

                val_accuracy = accuracy_score(ground_truth_label, predicted_label)
                print('Validation accuracy per round: ', val_accuracy)

                val_precision = precision_score(ground_truth_label, predicted_label, average='binary', zero_division=np.nan)
                print('Validation precision per round: ', val_precision)

                val_recall= recall_score(ground_truth_label, predicted_label, average='binary', zero_division=np.nan)
                print("Validation recall: ", val_recall)

                f_score = f1_score(ground_truth_label, predicted_label, average='binary', zero_division=np.nan)
                print('Validation f1 score per round: ', f_score)

                # Update the progress bar with the current epoch and loss
                vbar.set_description(f"Validation: {i}/{len(val_dataloader)}, Validation loss: {total_loss.item():.4f}")
                i+=1

    te_loss /= step
    val_accuracy = epoch_val_correct / epoch_val_total
    val_losses.append(te_loss)
    val_accuracies.append(val_accuracy)

    #Call the scheduler
    #scheduler.step(te_loss)

    #Convert lists of arrays to numpy arrays
    all_labels_array = np.concatenate(all_labels)
    all_preds_array = np.concatenate(all_preds)

    #Convert to 1D arrays
    all_labels_flat = all_labels_array.flatten()
    all_preds_flat = all_preds_array.flatten()

    #Ensure they are integers
    all_labels_int = all_labels_flat.astype(int)
    all_preds_int = all_preds_flat.astype(int)

    #Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels_int, all_preds_int)
    print("Confusion Matrix:")
    print(conf_matrix)

    #Get evaluation metrics:
    precision = precision_score(all_labels_int, all_preds_int, average='binary')
    recall = recall_score(all_labels_int, all_preds_int, average='binary')
    f_score= f1_score(all_labels_int, all_preds_int, average='binary')
    acc = accuracy_score(all_labels_int, all_preds_int)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f_score:.4f}")

    test_features, test_labels = get_features(test_dataloader)
    print('Test features created')
    visualize_features(test_features, test_labels, 'Test Features', epoch)

    #save best model and set the value of early stopping counter based loss
    if te_loss < best_te_loss:
        best_te_loss = te_loss
        best_ep = epoch
        #torch.save(model.state_dict(), "../fs_best_model_reducelr_s3.pt")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    #save last model
    print(f"epoch {epoch+1}, tr_loss {tr_loss}, te_loss {te_loss}")
    #torch.save(model.state_dict(), "../fs_last_model_reducelr_s3.pt")

    if (early_stopping_counter >= early_stopping_patience) or (best_te_loss<=early_stopping_threshold):
        print(f"Early stopping after {epoch + 1} epochs.")
        break

print(f"best epoch {best_ep+1}, best te_loss {best_te_loss}")
print('Start testing...')

#Testing model
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
        start_time = datetime.now()
        tbar = tqdm(test_dataloader, total=len(test_dataloader))
        i = 0
        for batch in tbar:
                # Extract images and texts from the batch
                images, true_label = batch 
                # Move images and texts to the specified device
                images = images.to(device)
                true_label = true_label.to(device)
                text_inputs = clip.tokenize(class_names).to(device)
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

                # Update the progress bar with the current epoch and loss
                tbar.set_description(f"Testing: {i}/{len(test_dataloader)}, Test loss: {total_loss.item():.4f}")
                i+=1

        end_time = datetime.now()
        print('Start time: ', start_time)
        print('Ending time: ', end_time)
        print('Overall time: ', end_time-start_time)
    
#Convert lists of arrays to numpy arrays
all_labels_array = np.concatenate(test_labels)
all_preds_array = np.concatenate(test_preds)

#Convert to 1D arrays
all_labels_flat = all_labels_array.flatten()
all_preds_flat = all_preds_array.flatten()

#Ensure they are integers
all_labels_int = all_labels_flat.astype(int)
all_preds_int = all_preds_flat.astype(int)

#Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels_int, all_preds_int)

#Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

#Get evaluation metrics:
precision = precision_score(all_labels_int, all_preds_int, average='binary')
recall = recall_score(all_labels_int, all_preds_int, average='binary')
f_score= f1_score(all_labels_int, all_preds_int, average='binary')
acc = accuracy_score(all_labels_int, all_preds_int)

#Print the metrics
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f_score:.4f}")

#Get number of CLIP model parameters (shouldn't change)
print("CLIP model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")

#Classification report
target_names = ['class 0', 'class 1']
print(classification_report(all_labels_int, all_preds_int , target_names=target_names))

#Plot the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig('training_validation_accuracy_5p.png')
plt.close()


#Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('training_validation_loss_5p.png')
plt.close()


