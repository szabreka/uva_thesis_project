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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import models
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.linear_model import LogisticRegression
import joblib

# Define device
if torch.cuda.is_available():
    device = torch.device("cuda") # use CUDA device
elif torch.backends.mps.is_available():
    device = torch.device("mps") # use MacOS GPU device (e.g., for M2 chips)
else:
    device = torch.device("cpu") # use CPU device
print('Used device: ', device)

class MobileNetV3Small_RNN(nn.Module):
    def __init__(self, num_classes, rnn_type="LSTM"):
        super(MobileNetV3Small_RNN, self).__init__()
        #load the model
        self.mobilenet = models.mobilenet_v3_small(weights = True)

        #freeze the mobilenet parameters (not training these for efficiency)
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        #extract features from final layer - pooling is exluded
        self.feature_extractor = self.mobilenet.features

        #Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        #get the number of features output by MobileNetV3 - for input for RNN
        self.num_features = self.mobilenet.classifier[0].in_features

        #room for rnn type choice: LSTM or GRU
        if rnn_type == "LSTM":
            #batch first to signal (B,C,F) format
            self.rnn = nn.LSTM(self.num_features, hidden_size=256, num_layers=1, batch_first=True)
        elif rnn_type == "GRU":
            #batch first to signal (B,C,F) format
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

#load cnn-rnn model    
rnn_model = MobileNetV3Small_RNN(num_classes=2, rnn_type="GRU")
state_dict = torch.load('../cnn_splits/light_cnn_best_model_gru_s3.pt', map_location=device)
state_dict = {k.partition('module.')[2] if k.startswith('module.') else k: v for k, v in state_dict.items()}
rnn_model.load_state_dict(state_dict)
rnn_model = rnn_model.to(device)
rnn_model.eval()

#load clip model
clip_model, preprocess = clip.load('ViT-B/16', device, jit=False)
state_dict = torch.load('../clip_splits/fs_last_model_s3_cosine.pt', map_location=device)
clip_model.load_state_dict(state_dict)
clip_model.eval()

#load logreg layer of CLIP-LIN
#logreg_model = joblib.load('../uva_thesis_project/final_logreg_model_best_61r.sav')

class ImageTitleDataset(Dataset):
    def __init__(self, list_video_path, list_labels, rnn_transform_image, clip_transform_image):
        #to handle the parent class
        super().__init__()
        #Initalize image paths and corresponding texts
        self.video_path = list_video_path
        #Initialize labels (0 or 1)
        self.labels = list_labels
        #Transform images based on defined transformation - for rnn 
        self.rnn_transform_image = rnn_transform_image
        #for clip
        self.clip_transform_image = clip_transform_image

    @staticmethod
    #Function to extract frames from video
    def preprocess_videos(video_path, num_rows=6, num_cols=6):
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
        
        return frames, concatenated_frames
    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #tranform videos into images and preprocess with defined transform function
        video_path = self.video_path[idx]
        #for cnn - frames
        frames, concatenated_frames = self.preprocess_videos(video_path)
        frames = [self.rnn_transform_image(Image.fromarray(frame)) for frame in frames]
        frames = torch.stack(frames)

        #clip
        image = Image.fromarray(concatenated_frames)
        image = self.clip_transform_image(image)

        #get the corresponding class names
        label = self.labels[idx]
        return frames, label, image
    


#Define training, validation and test data
def load_data(split_path):
    with open(split_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

#Date split
train_data = load_data('data/split/metadata_train_split_by_date.json')
val_data = load_data('data/split/metadata_validation_split_by_date.json')

#View splits
'''train_data = load_data('data/split/metadata_train_split_4_by_camera.json')
val_data = load_data('data/split/metadata_validation_split_4_by_camera.json')'''

# Prepare the list of video file paths and labels
train_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in train_data['file_name']]
train_list_labels = [int(label) for label in train_data['label']]
val_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in val_data['file_name']]
val_list_labels = [int(label) for label in val_data['label']]

#Define input resolution for cnn-rnn
rnn_input_resolution = (256, 256)

#Define the transformation pipeline for cnn-rnn
rnn_train_transform = transforms.Compose([
    transforms.Resize(rnn_input_resolution, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

rnn_val_test_transform = transforms.Compose([
    transforms.Resize(rnn_input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#Define class names in a list
class_names = ["a series picture of a factory with clear sky above chimney", "a series picture of a smoking factory"] #- 5

#Define input resolution for CLIP
input_resolution = (224, 224)

#Define the transformation pipeline for CLIP
clip_transform_steps = transforms.Compose([
    transforms.Resize(input_resolution, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
])

# Create dataset and data loader for training, validation and testing
train_dataset = ImageTitleDataset(train_list_video_path, train_list_labels, rnn_train_transform ,clip_transform_steps)
val_dataset = ImageTitleDataset(val_list_video_path, val_list_labels, rnn_val_test_transform, clip_transform_steps)
print('Datasets created')

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print('Dataloaders created')

#convert model function for clip
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

#Check if the device is set to CPU
if device == "cpu":
  clip_model.float()

class CLIP_MobileNetV3_RNN_Ensemble(nn.Module):
    def __init__(self, clip_model, mobilenet_rnn_model):
        super(CLIP_MobileNetV3_RNN_Ensemble, self).__init__()
        #set the models
        self.clip_model = clip_model
        self.mobilenet_rnn_model = mobilenet_rnn_model

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.mobilenet_rnn_model.parameters():
            param.requires_grad = False


    def forward(self, frames, image, classname):
        #send all data to device
        images = image.to(device)
        frames = frames.to(device)
        #get clip logits
        text_inputs = clip.tokenize(classname,context_length=77, truncate=True).to(device)
        #text_inputs.squeeze(dim=1)
        logits_per_image, logits_per_text = self.clip_model(images, text_inputs)

        #get mobilenet-rnn logits
        rnn_logits = self.mobilenet_rnn_model(frames)

        # Ensure data types for concatenation
        logits_per_image = logits_per_image.float()
        rnn_logits = rnn_logits.float()

        #combine logits
        combined_data = torch.cat((logits_per_image, rnn_logits), dim=1)

        return combined_data

class CLIP_GLIN_MobileNetV3_RNN_Ensemble(nn.Module):
    def __init__(self, clip_model, mobilenet_rnn_model, logreg_model):
        super(CLIP_GLIN_MobileNetV3_RNN_Ensemble, self).__init__()
        #set the models
        self.clip_model = clip_model
        self.mobilenet_rnn_model = mobilenet_rnn_model
        self.logreg_model = logreg_model

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.mobilenet_rnn_model.parameters():
            param.requires_grad = False


    def forward(self, frames, image, classname):
        #send all data to device
        images = image.to(device)
        frames = frames.to(device)
        #get clip logits
        features = self.clip_model.encode_image(images.to(device)).cpu().numpy()
        clip_glin_pred = self.logreg_model.predict_proba(features)
        clip_glin_pred = torch.tensor(clip_glin_pred).to(device)
        #to get the logits instead of the probabilites (this is the formula to capture logits)
        clip_glin_pred = torch.log(clip_glin_pred / (1 - clip_glin_pred))

        #get mobilenet-rnn logits
        rnn_logits = self.mobilenet_rnn_model(frames)

        # Ensure data types for concatenation
        clip_glin_pred = clip_glin_pred.float()
        rnn_logits = rnn_logits.float()

        #combine logits
        combined_data = torch.cat((clip_glin_pred, rnn_logits), dim=1)

        return combined_data

#ensemble with simple clip model
ensemble_model = CLIP_MobileNetV3_RNN_Ensemble(clip_model, rnn_model)
#ensemble with CLIP-LIN
#ensemble_model = CLIP_GLIN_MobileNetV3_RNN_Ensemble(clip_model, rnn_model,logreg_model)

#define test data
#test_data = load_data('data/split/metadata_test_split_4_by_camera.json')
test_data = load_data('data/split/metadata_test_split_by_date.json')

test_list_video_path = [os.path.join("/../projects/0/prjs0930/data/merged_videos/", f"{fn}.mp4") for fn in test_data['file_name']]
test_list_labels = [int(label) for label in test_data['label']]
test_dataset = ImageTitleDataset(test_list_video_path, test_list_labels, rnn_val_test_transform, clip_transform_steps)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#get features
def get_features(model, dataloader):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for frames, label, image in dataloader:
            
            features = model(frames, image, class_names)
            all_features.extend(features.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    
    return all_features, all_labels

val_features, val_labels = get_features(ensemble_model, val_dataloader)
meta_model = LogisticRegression()
meta_model.fit(val_features, val_labels)

#evaluate model
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
            
            test_preds, test_labels = get_features(model, dataloader)
            meta_preds = meta_model.predict(test_preds)
            
            all_preds.extend(meta_preds)
            all_labels.extend(test_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = evaluate_model(ensemble_model, test_dataloader, device)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

#get number of parameters in model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#Count parameters of the RNN model
print("Number of trainable parameters in RNN model: ", count_parameters(rnn_model))

#Count parameters of the CLIP model
print("Number of trainable parameters in CLIP model: ", count_parameters(clip_model))

#Count parameters of the CLIP model
print("Number of trainable parameters in Logreg model: ", meta_model.coef_.size)

#Count parameters of the ensemble model
ensemble_model = CLIP_MobileNetV3_RNN_Ensemble(clip_model, rnn_model)
print("Number of trainable parameters in ensemble model: ", count_parameters(ensemble_model))

#save logreg model
filename = '../logreg_model_ensemble_stacking_lstm_s3.sav'
joblib.dump(meta_model, filename)