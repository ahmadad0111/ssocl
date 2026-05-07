
import os
import scipy.io as sio

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
class DEAPDatasetLoader:
    def __init__(self,args, data_path='./data/cross_sub/DEAP/', sampling_rate=128, segment_size=768, stride=128, split_ratio=0.8):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.stride = stride
        self.split_ratio = split_ratio
        self.args = args
        self.global_mean = None
        self.global_std = None

    # def categorize_sample(self, val, arousal, label_mode):
    #     if label_mode == 'VA':
    #         if val >= 5.5 and arousal >= 5.5:
    #             return 0
    #         elif val >= 5.5 and arousal < 4.5:
    #             return 1
    #         elif val < 4.5 and arousal < 4.5:
    #             return 2
    #         elif val < 4.5 and arousal >= 5.5:
    #             return 3
    #         else:
    #             return 4  # Undefined class for filtering
    #     elif label_mode == 'V':
    #         return 0 if val >= 5 else 1
    #     elif label_mode == 'A':
    #         return 0 if arousal >= 5 else 1


    def categorize_sample(self, val, arousal, label_mode):
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

    def load_and_normalize_data(self, label_mode='VA'):
        all_data, all_labels = [], []
        num_subjects = 32  # Total subjects in DEAP dataset
        remove_few_sec = 3 + 10  # Remove first 3s + 10s baseline

        # Load all subjects' data
        for subject_id in range(1, num_subjects + 1):
            data_file = f'{self.data_path}s{str(subject_id).zfill(2)}.dat'
            
            with open(data_file, 'rb') as f:
                dataset = pickle.load(f, encoding='latin1')
            
            labels = dataset['labels']
            eeg_data = dataset['data'][:, :32, remove_few_sec * self.sampling_rate:]  # Shape: (40 trials, 32 channels, time)

            for i in range(eeg_data.shape[0]):  # Iterate over trials
                trial_data = eeg_data[i]
                val, arousal = labels[i][0], labels[i][1]
                label = self.categorize_sample(val, arousal, label_mode)
                all_data.append(trial_data)  # Append raw trial data
                all_labels.append(label)  # Append trial label

        # Convert lists to numpy arrays
        all_data = np.array(all_data).astype(np.float32)  # Shape: (subjects * trials, channels, time)
        all_labels = np.array(all_labels, dtype=np.float32)  # Shape: (subjects * trials,)

        # Filter out undefined labels (only for 'VA' mode)
        if label_mode == 'VA':
            mask = all_labels != 4
            all_data = all_data[mask]
            all_labels = all_labels[mask]

        # Global normalization: Compute mean and std across all trials, channels, and time
        global_mean = all_data.mean()
        global_std = all_data.std()
        all_data = (all_data - global_mean) / (global_std + 1e-8)  # Normalize globally

        return  global_mean, global_std

    def load_subject_data(self, label_mode='VA', subject_id=1):
        if self.global_mean is None or self.global_std is None:
            self.global_mean, self.global_std = self.load_and_normalize_data(label_mode)

        all_data, all_labels = [], []

        num_subjects = 32  # Assuming there are 32 subjects in the DEAP dataset
        remove_few_sec = 3 + 10

        #for subject_id in range(1, num_subjects + 1):
        data_file = f'{self.data_path}s{str(subject_id).zfill(2)}.dat'
        
        with open(data_file, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        
        labels = dataset['labels']
        eeg_data = dataset['data'][:, :32, remove_few_sec * self.sampling_rate:]
        
        num_segments = (eeg_data.shape[2] - self.segment_size) // self.stride + 1

        for i in range(eeg_data.shape[0]):
            trial_data = eeg_data[i]
            val, arousal = labels[i][0], labels[i][1]
            label = self.categorize_sample(val, arousal, label_mode)

            trial_segments, trial_labels = [], []
            for j in range(num_segments):
                start, end = j * self.stride, j * self.stride + self.segment_size
                segment = trial_data[:, start:end]
                trial_segments.append(segment)
                trial_labels.append(label)

            # Calculate split point
            split_point = int(len(trial_segments) * self.split_ratio)

            # Append train segments and labels
            all_data.extend(trial_segments)
            all_labels.extend(trial_labels)
            
        
        # Convert lists to numpy arrays
        all_data = np.array(all_data).astype(np.float32)
        all_labels = np.array(all_labels, dtype=np.float32)
        
        # Filter out undefined labels (only for 'VA' mode)
        if label_mode == 'VA':
            all_mask = all_labels != 4
            all_data = all_data[all_mask]
            all_labels = all_labels[all_mask]


        # Replace NaNs with zeros in both train and test data
        all_data[np.isnan(all_data)] = 0


        all_data = (all_data - self.global_mean) / (self.global_std + 1e-8)  # Normalize globally


        if len(all_data) < 10:
            return None, None, None, None
        
        # Split data into train and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

        return train_data, train_labels, test_data, test_labels


    def load_all_deap_data(self, label_mode='VA', num_subjects=32, channels = 32):
        if self.global_mean is None or self.global_std is None:
            self.global_mean, self.global_std = self.load_and_normalize_data(label_mode)
        all_data, all_labels = [], []

        #num_subjects = 32  # Assuming there are 32 subjects in the DEAP dataset
        remove_few_sec = 3 + 10

        for subject_id in range(1, num_subjects):
            data_file = f'{self.data_path}s{str(subject_id).zfill(2)}.dat'
            
            with open(data_file, 'rb') as f:
                dataset = pickle.load(f, encoding='latin1')
            
            labels = dataset['labels']
            eeg_data = dataset['data'][:, :32, remove_few_sec * self.sampling_rate:]
            
            num_segments = (eeg_data.shape[2] - self.segment_size) // self.stride + 1

            for i in range(eeg_data.shape[0]):
                trial_data = eeg_data[i]
                val, arousal = labels[i][0], labels[i][1]
                label = self.categorize_sample(val, arousal, label_mode)

                trial_segments, trial_labels = [], []
                for j in range(num_segments):
                    start, end = j * self.stride, j * self.stride + self.segment_size
                    segment = trial_data[:, start:end]
                    trial_segments.append(segment)
                    trial_labels.append(label)


                
                # Append train segments and labels
                all_data.extend(trial_segments)
                all_labels.extend(trial_labels)

        # Convert lists to numpy arrays
        all_data = np.array(all_data).astype(np.float32)
        # all_data = all_data[:,0:channels]
        all_labels = np.array(all_labels, dtype=np.float32)
        
        # Filter out undefined labels (only for 'VA' mode)
        if label_mode == 'VA':
            all_mask = all_labels != 4
            all_data = all_data[all_mask]
            all_labels = all_labels[all_mask]


        # Replace NaNs with zeros in both train and test data
        all_data[np.isnan(all_data)] = 0


        all_data = (all_data - self.global_mean) / (self.global_std + 1e-8)  # Normalize globally

        if channels < 32:
            #all_data = all_data[:,0:channels]
            if self.args.dataset == 'AMIGOS':
                print('DEAP to AMIGOS channels are being selected')
                deap_to_amigos_indices = [1, 3, 2, 4, 7, 11, 13, 31, 29, 25, 21, 19, 20, 17]
                all_data = all_data[:,deap_to_amigos_indices,:]





        # Split data into train and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        return  train_data, test_data, train_labels, test_labels



class AMIGOSDatasetLoader:
    def __init__(self,args, data_path='./data/cross_sub/DEAP/', sampling_rate=128, segment_size=768, stride=128, split_ratio=0.8):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.stride = stride
        self.split_ratio = split_ratio
        self.args = args

        self.global_mean = None
        self.global_std = None

    def segment_data(self,data, segment_length_sec, stride_rate=0.5, sampling_rate=128):
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

    def categorize_val_arousal_self(self,label):
        val = label[0][0]
        arousal = label[0][1]
        if val >= 5.5 and arousal >= 5.5:
            return 0
        elif val >= 5.5 and arousal < 4.5:
            return 1
        elif val < 4.5 and arousal < 4.5:
            return 2
        elif val < 4.5 and arousal >= 5.5:
            return 3
        else:
            return 4  # Undefined class for filtering
    # def categorize_val_arousal_self(self,label):
    #     val = label[0][0]
    #     arousal = label[0][1]
    #     if val >= 7 and arousal >= 7:
    #         return 0
    #     elif val >= 7 and arousal < 3:
    #         return 1
    #     elif val < 3 and arousal < 3:
    #         return 2
    #     elif val < 3 and arousal >= 7:
    #         return 3
    #     else:
    #         return 4  # Undefined class for filtering


    def load_and_normalize_data(self, label_mode='VA'):
        self_annotation = True

        sampling_rate = 128  # Hz
        segment_length_sec = 7  # seconds
        stride_rate =  0.5#0.5 # 50% overlap


        # get data and labels for all video trial
        segmented_data_all = []
        categorized_labels_all = []


        for subject_id in range(1,41):

            if subject_id in [8,24,28,32]:#[26,31,4,5,10,11,22,25,28,30,40,8,24,32,27]: #]
                continue

            combined_data = self.load_data_mat(subject_id)
            for video_idx in range(combined_data['joined_data'].shape[1] ):
                data = combined_data['joined_data'][0,video_idx]
                labels_self = combined_data['labels_selfassessment'][0,video_idx][:,0:2]
                labels_self = self.categorize_val_arousal_self(labels_self)
                #labels_ext_annot = combined_data['labels_ext_annotation'][0,video_idx]

                # Segment the data with zero-padding
                segmented_data = self.segment_data(data, segment_length_sec, stride_rate, sampling_rate)
                segmented_labels = np.repeat(np.array(labels_self),len(segmented_data),axis=0)


                segmented_data_all.extend(segmented_data)
                categorized_labels_all.extend(segmented_labels)


        # now separate EEG, ECG and GSR data
        frames_EEG = []
        frames_ECG = []
        frames_GSR = []

        for segment in segmented_data_all:
            frames_EEG.append(segment[:,0:14])
            frames_ECG.append(segment[:,14:16])
            frames_GSR.append(segment[:,-1])


        # convert into arrays and swap last two dimensions
        frames_EEG = np.array(frames_EEG)
        frames_ECG = np.array(frames_ECG)
        frames_GSR = np.array(frames_GSR)
        # replace nan with zeros
        frames_EEG[np.isnan(frames_EEG)] = 0
        frames_ECG[np.isnan(frames_ECG)] = 0
        frames_GSR[np.isnan(frames_GSR)] = 0

        frames_EEG = np.swapaxes(frames_EEG, 1, 2)
        frames_ECG = np.swapaxes(frames_ECG, 1, 2)

        # convert to float32
        frames_EEG = frames_EEG.astype(np.float32)
        frames_ECG = frames_ECG.astype(np.float32)
        frames_GSR = frames_GSR.astype(np.float32)

        # convert labels to float32
        categorized_labels_all = np.array(categorized_labels_all)
        categorized_labels_all = categorized_labels_all.astype(np.float32)

        # Filter out undefined labels (only for 'VA' mode)
        if label_mode == 'VA':
            mask = categorized_labels_all != 4
            all_data = frames_EEG[mask]
            all_labels = categorized_labels_all[mask]

        # Global normalization: Compute mean and std across all trials, channels, and time
        global_mean = all_data.mean()
        global_std = all_data.std()
        all_data = (all_data - global_mean) / (global_std + 1e-8)  # Normalize globally

        return  global_mean, global_std

    def load_subject_data(self, label_mode='VA', subject_id=1):
        if self.global_mean is None or self.global_std is None:
            self.global_mean, self.global_std = self.load_and_normalize_data(label_mode)


        sampling_rate = 128  # Hz
        segment_length_sec = 7  # seconds
        stride_rate =  0.5 # 50% overlap

        # get data and labels for all video trial
        segmented_data_all = []
        categorized_labels_all = []
            
        combined_data = self.load_data_mat(subject_id)
        for video_idx in range(combined_data['joined_data'].shape[1] ):
            data = combined_data['joined_data'][0,video_idx]
            labels_self = combined_data['labels_selfassessment'][0,video_idx][:,0:2]
            labels_self = self.categorize_val_arousal_self(labels_self)
            #labels_ext_annot = combined_data['labels_ext_annotation'][0,video_idx]

            # Segment the data with zero-padding
            segmented_data = self.segment_data(data, segment_length_sec, stride_rate, sampling_rate)
            segmented_labels = np.repeat(np.array(labels_self),len(segmented_data),axis=0)


            segmented_data_all.extend(segmented_data)
            categorized_labels_all.extend(segmented_labels)


        # now separate EEG, ECG and GSR data
        frames_EEG = []
        frames_ECG = []
        frames_GSR = []

        for segment in segmented_data_all:
            frames_EEG.append(segment[:,0:14])
            frames_ECG.append(segment[:,14:16])
            frames_GSR.append(segment[:,-1])


        # convert into arrays and swap last two dimensions
        frames_EEG = np.array(frames_EEG)
        frames_ECG = np.array(frames_ECG)
        frames_GSR = np.array(frames_GSR)
        # replace nan with zeros
        frames_EEG[np.isnan(frames_EEG)] = 0
        frames_ECG[np.isnan(frames_ECG)] = 0
        frames_GSR[np.isnan(frames_GSR)] = 0

        frames_EEG = np.swapaxes(frames_EEG, 1, 2)
        frames_ECG = np.swapaxes(frames_ECG, 1, 2)

        # convert to float32
        frames_EEG = frames_EEG.astype(np.float32)
        frames_ECG = frames_ECG.astype(np.float32)
        frames_GSR = frames_GSR.astype(np.float32)

        # convert labels to float32
        categorized_labels_all = np.array(categorized_labels_all)
        categorized_labels_all = categorized_labels_all.astype(np.float32)

        # Filter out undefined labels (only for 'VA' mode)
        if label_mode == 'VA':
            mask = categorized_labels_all != 4
            all_data = frames_EEG[mask]
            all_labels = categorized_labels_all[mask]

        all_data = (all_data - self.global_mean) / (self.global_std + 1e-8)  # Normalize globally


        if len(all_data) < 10:
            return None, None, None, None
        
        # Split data into train and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

        return train_data, train_labels, test_data, test_labels


    def load_all_deap_data(self, label_mode='VA', num_subjects=41):
        if self.global_mean is None or self.global_std is None:
            self.global_mean, self.global_std = self.load_and_normalize_data(label_mode)


        sampling_rate = 128  # Hz
        segment_length_sec = 7  # seconds
        stride_rate =  0.5 # 50% overlap

        # get data and labels for all video trial
        segmented_data_all = []
        categorized_labels_all = []


        for subject_id in range(1,41):

            # if subject_id in [26,31,4,5,10,11,22,25,28,30,40,8,24,32,27]: #]
            #     continue

            if subject_id in [8,24,28,32]: 
                continue

            
            combined_data = self.load_data_mat(subject_id)
            for video_idx in range(combined_data['joined_data'].shape[1]):
                data = combined_data['joined_data'][0,video_idx]
                labels_self = combined_data['labels_selfassessment'][0,video_idx][:,0:2]
                labels_self = self.categorize_val_arousal_self(labels_self)
                #labels_ext_annot = combined_data['labels_ext_annotation'][0,video_idx]

                # Segment the data with zero-padding
                segmented_data = self.segment_data(data, segment_length_sec, stride_rate, sampling_rate)
                segmented_labels = np.repeat(np.array(labels_self),len(segmented_data),axis=0)


                segmented_data_all.extend(segmented_data)
                categorized_labels_all.extend(segmented_labels)


        # now separate EEG, ECG and GSR data
        frames_EEG = []
        frames_ECG = []
        frames_GSR = []

        for segment in segmented_data_all:
            frames_EEG.append(segment[:,0:14])
            frames_ECG.append(segment[:,14:16])
            frames_GSR.append(segment[:,-1])


        # convert into arrays and swap last two dimensions
        frames_EEG = np.array(frames_EEG)
        frames_ECG = np.array(frames_ECG)
        frames_GSR = np.array(frames_GSR)
        # replace nan with zeros
        frames_EEG[np.isnan(frames_EEG)] = 0
        frames_ECG[np.isnan(frames_ECG)] = 0
        frames_GSR[np.isnan(frames_GSR)] = 0

        frames_EEG = np.swapaxes(frames_EEG, 1, 2)
        frames_ECG = np.swapaxes(frames_ECG, 1, 2)

        # convert to float32
        frames_EEG = frames_EEG.astype(np.float32)
        frames_ECG = frames_ECG.astype(np.float32)
        frames_GSR = frames_GSR.astype(np.float32)

        # convert labels to float32
        categorized_labels_all = np.array(categorized_labels_all)
        categorized_labels_all = categorized_labels_all.astype(np.float32)

        # Filter out undefined labels (only for 'VA' mode)
        if label_mode == 'VA':
            mask = categorized_labels_all != 4
            all_data = frames_EEG[mask]
            all_labels = categorized_labels_all[mask]

        all_data = (all_data - self.global_mean) / (self.global_std + 1e-8)  # Normalize globally

        # Split data into train and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        return  train_data, test_data, train_labels, test_labels

    def load_data_mat(self,subject_id):
        data_folder='C:/Users/adnan/Desktop/AMIGOS/retain_adapt_AMBM_model/data/cross_sub/Data_Preprocessed_P' + str(subject_id).zfill(2) + '/Data_Preprocessed_P' + str(subject_id).zfill(2) + '.mat'
        data = sio.loadmat(data_folder)
        return data