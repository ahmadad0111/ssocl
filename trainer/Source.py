import numpy as np
import math
import time
import scipy
import numpy as np
import math
import time
import scipy
#import scipy.signal
import scipy.io
# import self defined functions
#from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
#from scipy import interp

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import scipy.signal
import scipy.io
# import self defined functions 
from torch.utils.data import Dataset
import random
import scipy.io as sio
from scipy import interp
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from models import EEGNet1D

from data import DEAPDatasetLoader,AMIGOSDatasetLoader

class SourceTrainer(object):
    def __init__(self,args):

        self.args = args

        self.set_seed(self.args.seed)

        self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_32.pth'

    def train(self):

        if self.args.source_data == 'PPB_EMO':

            if self.args.dataset == 'AMIGOS':
                self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_14.pth'
                model = EEGNet1D(in_channels = 14)

                if self.args.multisource == True:
                    model.load_state_dict(torch.load(self.best_model_path))

                train_loader, val_loader = self.ppbemo_data_loaders(channels = 14)
                self.source_training(train_loader,val_loader, model)               

            else:
                self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_32.pth'
                model = EEGNet1D(in_channels = 32)

                train_loader, val_loader = self.ppbemo_data_loaders(channels = 32)
                self.source_training(train_loader,val_loader, model)
        elif self.args.source_data == 'DEAP':

            if self.args.dataset == 'AMIGOS':
                self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_14.pth'
                model = EEGNet1D(in_channels = 14)

                if self.args.multisource == True:
                    model.load_state_dict(torch.load(self.best_model_path))

                deap_data = DEAPDatasetLoader(self.args)

                X_train, X_test, y_train, y_test = deap_data.load_all_deap_data(channels = 14)

                # Convert to torch tensors
                X_train_tensor = torch.tensor(X_train)
                y_train_tensor = torch.tensor(y_train)
                X_val_tensor = torch.tensor(X_test)
                y_val_tensor = torch.tensor(y_test)

                # Create a dataloader
                train_data = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

                val_data = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
                self.source_training(train_loader,val_loader, model)   
             

            else:
                self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_32.pth'
                model = EEGNet1D(in_channels = 32)

                deap_data = DEAPDatasetLoader(self.args)

                X_train, X_test, y_train, y_test = deap_data.load_all_deap_data()

                # Convert to torch tensors
                X_train_tensor = torch.tensor(X_train)
                y_train_tensor = torch.tensor(y_train)
                X_val_tensor = torch.tensor(X_test)
                y_val_tensor = torch.tensor(y_test)

                # Create a dataloader
                train_data = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

                val_data = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
                self.source_training(train_loader,val_loader, model)   
             
        elif self.args.source_data == 'AMIGOS':

            if self.args.dataset == 'AMIGOS':
                self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_14.pth'
                model = EEGNet1D(in_channels = 14)

                if self.args.multisource == True:
                    model.load_state_dict(torch.load(self.best_model_path))

                deap_data = AMIGOSDatasetLoader(self.args)

                X_train, X_test, y_train, y_test = deap_data.load_all_deap_data()

                # Convert to torch tensors
                X_train_tensor = torch.tensor(X_train)
                y_train_tensor = torch.tensor(y_train)
                X_val_tensor = torch.tensor(X_test)
                y_val_tensor = torch.tensor(y_test)

                # Create a dataloader
                train_data = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

                val_data = TensorDataset(X_val_tensor, y_val_tensor)
                val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
                self.source_training(train_loader,val_loader, model)   
             

            else:
                # exception error data not supported
                raise ValueError('Data not supported')
               





    def source_training(self, train_loader,val_loader, model):


        # Lists to store the history of losses and accuracies
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        best_val_loss = np.inf

        wait = 0
        #print(model)
        model = model.to(device= self.args.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # Define the learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)



        # Train the model
        model.train()
        for epoch in range(self.args.source_epochs):
            train_loss = 0.0
            train_accuracy = 0.0
            all_labels = []
            all_predictions = []
            for data, target in train_loader:
                data = data.to(self.args.device)
                target = target.to(self.args.device)
                optimizer.zero_grad()

                #print(data.size())
                output,z = model(data)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)

                # Get predictions
                _, predicted = torch.max(output, 1)
                all_labels.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())


            epoch_loss = train_loss/len(train_loader.dataset)
            epoch_acc = accuracy_score(all_labels, all_predictions)
            train_loss_history.append(epoch_loss)
            train_acc_history.append(epoch_acc)

            val_loss, val_acc = self.validate_model(model, val_loader,criterion)

                # Store the validation loss and accuracy
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc.item())
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                torch.save(model.state_dict(), self.best_model_path)  # Save the best model
                print(f"Epoch {epoch+1}/{self.args.source_epochs} | Validation loss improved. Model saved.")
            else:
                wait += 1
                print(f"Epoch {epoch+1}/{self.args.source_epochs} | Validation loss did not improve.")

        # Early stopping
            if wait >= self.args.source_patience:
                print("Early stopping triggered. No improvement in validation loss for", self.args.source_patience, "epochs.")
                break

            print(f'Epoch [{epoch+1}/{self.args.source_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

            #print('Epoch: {} \tTraining Loss: {:.6f} \t Training Acc: {}'.format(epoch+1, train_loss,accuracy))
        
        
    def set_seed(self,seed):
        # Set Python, NumPy, and PyTorch seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Ensure reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def ppbemo_data_loaders(self, channels = 32):


        # get data

        video_id = ['AD', 'HD', 'SD', 'DD', 'ND', 'FD','SAD']

        subject_id = 2

        sampling_rate = 250  # Hz
        segment_length_sec = 2  # seconds
        stride_rate =  0.5 # 50% overlap

        # get data and labels for all video trial
        segmented_data_all = []
        categorized_labels_all = []

        # load labels
        all_labels_csv = self.load_labels_emo()
        for subject_id in range(2, 41):

            for video_id_s in video_id:
                data, video_id_s = self.load_data_ppb_emo(subject_id, video_id_s)

                if data is None:
                    continue

                start_index = max(0, data.shape[1] - 32)
                data = data.iloc[:, start_index:]


                label = all_labels_csv[all_labels_csv['PPB_Emo_dataset@Physiological_data'] == video_id_s]
                label_catg = self.categorize_sample_ppb_emo(label['valence'].values[0], label['arousal'].values[0])

                # data segmentation
                segmented_data = self.segment_data_ppb_emo(data, segment_length_sec, stride_rate, sampling_rate)
                segmented_labels = np.ones(segmented_data.shape[0]) * label_catg

                segmented_data_all.extend(segmented_data)
                categorized_labels_all.extend(segmented_labels)



        # data numpy
        segmented_data_all = np.array(segmented_data_all)

        ppb_to_deap_indices = [ 0, 2, 5, 4, 9, 10, 14, 13, 18, 19, 23, 22, 27, 29, 30, 24, 1, 3, 6, 7, 8, 12, 11, 15, 16, 17, 21, 20, 25, 26, 28, 31]


        segmented_data_all = segmented_data_all[:,:,ppb_to_deap_indices]

        # segmented_data_all = segmented_data_all[:,:,0:channels]


        if channels < 32:
            #all_data = all_data[:,0:channels]
            if self.args.dataset == 'AMIGOS':
                print('DEAP to AMIGOS channels are being selected')
                deap_to_amigos_indices = [1, 3, 2, 4, 7, 11, 13, 31, 29, 25, 21, 19, 20, 17]
                segmented_data_all = segmented_data_all[:,:,deap_to_amigos_indices]

        # replace nan with 0
        segmented_data_all = np.nan_to_num(segmented_data_all)





        # normalize data
        scaler = StandardScaler()
        #print(segmented_data_all.shape)
        segmented_data_all = scaler.fit_transform(segmented_data_all.reshape(-1, segmented_data_all.shape[-1])).reshape(segmented_data_all.shape)
        # swap axes 1,2
        #print(segmented_data_all.shape)
        segmented_data_all = np.swapaxes(segmented_data_all, 1, 2)

        # convert to float32
        segmented_data_all = segmented_data_all.astype(np.float32)
        # convert label to float32
        categorized_labels_all = np.array(categorized_labels_all).astype(np.float32)

        # Filter out undefined labels (only for 'VA' mode)

        all_mask = categorized_labels_all != 4
        segmented_data_all = segmented_data_all[all_mask]
        categorized_labels_all = categorized_labels_all[all_mask]



        # split data
        X_train, X_test, y_train, y_test = train_test_split(segmented_data_all, categorized_labels_all, test_size=0.2, random_state=42, stratify=categorized_labels_all)


        # shape of data

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train)
        y_train_tensor = torch.tensor(y_train)
        X_val_tensor = torch.tensor(X_test)
        y_val_tensor = torch.tensor(y_test)

        # Create a dataloader
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        val_data = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

        return train_loader, val_loader





    def load_data_ppb_emo(self,subject_id, video_id):
        data_folder=r'C:\Users\adnan\Desktop\ppb_emo\Physiological_data\P' + str(subject_id).zfill(2) + '/PPB_Emo_dataset@EEG-30s-P' + str(subject_id).zfill(2)+'-' +video_id + '.csv'

        # Check if the file exists
        if os.path.exists(data_folder):

            df = pd.read_csv(data_folder)
            video_id_ = 'PPB_Emo_dataset@EEG-30s-P' + str(subject_id).zfill(2)+'-' +video_id
            return df, video_id_
        else:
            return None, None


    def segment_data_ppb_emo(self,data, segment_length_sec, stride_rate=0.5, sampling_rate=128):
        segment_length = segment_length_sec * sampling_rate  # Convert segment length from seconds to samples
        stride = int(segment_length * stride_rate)  # Calculate stride length in samples

        # Calculate how much zero-padding is needed
        pad_length = segment_length - (len(data) % segment_length)

        # Pad data with zeros to ensure it can be evenly divided into segments
        if pad_length > 0:
            data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')

        segments = []
        for start in range(0, len(data) - segment_length + 1, stride):
            segment = data[start:start + segment_length]
            segments.append(segment)

        return np.array(segments)


    def load_labels_emo(self):
        data_folder='./data/cross_sub/ppb_emo/EE Data/Emotion_label.xlsx'
        return pd.read_excel(data_folder, engine='openpyxl')
    
    def categorize_sample_ppb_emo(self,val, arousal, label_mode = 'VA'):
        if label_mode == 'VA':
            if val >= 7 and arousal >= 7:
                return 0
            elif val >= 7 and arousal < 3:
                return 1
            elif val < 3 and arousal < 3:
                return 2
            elif val < 3 and arousal >= 7:
                return 3
            else:
                return 4  # Undefined class for filtering
        elif label_mode == 'V':
            return 0 if val >= 5 else 1
        elif label_mode == 'A':
            return 0 if arousal >= 5 else 1
        

    def validate_model(self,model, val_loader,criterion):
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                # Forward pass
                outputs,z = model(inputs)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

        val_loss /= total
        val_acc = correct.double() / total
        return val_loss, val_acc
