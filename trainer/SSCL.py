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
import csv
from itertools import zip_longest
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
from models import EEGNet1D, PredictorNet
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*")
from data import DEAPDatasetLoader, AMIGOSDatasetLoader
from buffer import Buffer
import copy

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from torch.utils.data import Sampler
class MultiLabelDataset(Dataset):
    def __init__(self, data, labels):
        # Store the data and labels
        self.data = data
        self.targets = labels  # Unpack the two labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve data and both labels by index
        sample = self.data[idx]
        target = self.targets[idx]

        
        # Return data and both labels
        return sample, target
    


# Two classes in each batch

class SeqSampler(Sampler):
    def __init__(self, dataset, blend_ratio, n_concurrent_classes, batch_size):
        """data_source is a Subset"""
        self.num_samples = len(dataset)
        self.blend_ratio = blend_ratio
        self.n_concurrent_classes = n_concurrent_classes
        self.batch_size = batch_size

        # Configure the correct train_subset and val_subset
        if torch.is_tensor(dataset.targets):
            self.labels = dataset.targets.detach().cpu().numpy()
        else:  # targets in CIFAR10 and CIFAR100 are lists
            self.labels = np.array(dataset.targets)
        self.classes = list(set(self.labels))
        self.n_classes = len(self.classes)

        # Sort classes
        self.classes.sort()

    def __iter__(self):
        """Sequential sampler"""
        # Generate indices for all classes
        sample_idx = []
        for c in self.classes:
            # Filter samples for the current class
            filtered_ind = np.where(self.labels == c)[0]  # Indices of the samples of class `c`
            np.random.shuffle(filtered_ind)  # Shuffle the samples within the class
            sample_idx.append(filtered_ind.tolist())

        # Create batches containing two classes each
        final_idx = []
        while len(final_idx) < self.num_samples:
            # Randomly select two different classes
            class_pair = random.sample(self.classes, 2)
            # Get indices for these two classes
            class1_idx = sample_idx[self.classes.index(class_pair[0])]
            class2_idx = sample_idx[self.classes.index(class_pair[1])]

            # Create batches of `batch_size // 2` for each class
            num_samples_per_class = self.batch_size // 2
            batch_class1 = class1_idx[:num_samples_per_class]
            batch_class2 = class2_idx[:num_samples_per_class]
            final_idx.append(batch_class1 + batch_class2)

            # Remove the used indices from both class lists
            sample_idx[self.classes.index(class_pair[0])] = class1_idx[num_samples_per_class:]
            sample_idx[self.classes.index(class_pair[1])] = class2_idx[num_samples_per_class:]

        # Flatten the list of batches into one final list of indices
        final_idx = [item for sublist in final_idx for item in sublist]
        return iter(final_idx)

    def __len__(self):
        # Return the total number of batches
        return self.num_samples // self.batch_size




class SSCLTrainer(object):
    def __init__(self,args):

        self.args = args

        self.set_seed(self.args.seed)

        self.best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_32.pth'
        self.criterion = nn.CrossEntropyLoss()
        

    def train(self):

        self.buffer_batch_size = self.args.buffer_batch_size
        buffer = Buffer(buffer_size=self.args.buffer_size,device=self.args.device)

        if self.args.dataset == 'DEAP':
            best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_32.pth'
            model = EEGNet1D(in_channels = 32)
            model = model.to(device = self.args.device)
            # load the best model
            model.load_state_dict(torch.load(best_model_path))

            # load data
            self.deap_loader = DEAPDatasetLoader(self.args)

            _,test_gen_data,_, test_gen_labels = self.deap_loader.load_all_deap_data()
            test_gen_dataset = MultiLabelDataset(test_gen_data, test_gen_labels)
            test_gen_loader = DataLoader(test_gen_dataset, batch_size=self.args.batch_size, shuffle=True)

        elif self.args.dataset == 'AMIGOS':
            best_model_path = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\best_model_EEG_14.pth'

            model = EEGNet1D(in_channels = 14)
            model = model.to(device = self.args.device)
            # load the best model
            model.load_state_dict(torch.load(best_model_path))

            self.deap_loader = AMIGOSDatasetLoader(self.args)        
       
            _,test_gen_data,_, test_gen_labels = self.deap_loader.load_all_deap_data()
            test_gen_dataset = MultiLabelDataset(test_gen_data, test_gen_labels)
            test_gen_loader = DataLoader(test_gen_dataset, batch_size=self.args.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=self.args.base_lr)

        # Define the learning rate scheduler
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # Train the model


        model.train()

        sub_ids = []
        adaptation_accuracy = []
        generalization_accuracy = []
        forgetting_mitigation_accuracy = []

        buffer_emb_display = []
        buffer_label_display = []
        sub_emb_display = []
        sub_label_display = []

        subject_ids = list(range(1, self.args.subject_range))

        # Shuffle the subject IDs to randomize the order
        random.shuffle(subject_ids)

        for subject_id in range(1, self.args.subject_range):
            #for subject_id in subject_ids:

            if self.args.dataset == 'AMIGOS':
                if subject_id in [8,24,28,32]:# [27,26,31,4,5,10,11,22,25,28,30,40,8,24,32]:
                    continue

            print(f"Training model for subject {subject_id}...")
            if subject_id > 1:
                train_loader, val_loader, test_for_loader = self.data_loader(subject_id=subject_id, batch_size=self.args.batch_size)
            else:
                train_loader, val_loader = self.data_loader(subject_id=subject_id, batch_size=self.args.batch_size)

            if train_loader is None:
                print(f"No data found for subject {subject_id}. Skipping...")
                continue

            for epoch in range(self.args.n_epochs):
                train_loss = 0.0
                train_accuracy = 0.0
                all_labels = []
                all_predictions = []
                for batch_data, batch_labels in train_loader:

                    data = batch_data
                    target = batch_labels

                    if len(data) < 10:
                        continue

                    data = data.to(self.args.device)
                    target = target.to(self.args.device)

                    #  pseudo labels to the buffer via cpc  and kmeans clustering

                    if not buffer.is_empty():
                        #print("Buffer is not empty")
                        buff_data, buff_labels = buffer.get_all_data()
                        data_comb = torch.cat([data, buff_data], dim=0)
                        personalized_model = self.cpc_training(copy.deepcopy(model),self.args.device,data)
                    else:
                        #print("Buffer is empty")
                        personalized_model = self.cpc_training(copy.deepcopy(model),self.args.device,data)

                    _,embeddings = personalized_model(data)

                    embeddings = F.normalize(embeddings, p=2, dim=1)

                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=1000)
                    pseudo_labels = kmeans.fit_predict(embeddings.cpu().detach().numpy())

                    
                    # check if memory buffer is empty
                    if buffer.is_empty():
                        #print("Buffer is empty adding data")
                        buffer.add_data(examples=data, labels=torch.tensor(pseudo_labels).to(self.args.device))

                        data_cluster = data
                        pseudo_labels_cluster = torch.tensor(pseudo_labels).to(self.args.device)
                    else:
                        # get all the data from the buffer
                        buf_data, buf_labels = buffer.get_all_data()

                        pseudo_labels = torch.tensor(pseudo_labels).to(self.args.device)

                        data_cluster,pseudo_labels_cluster, data_to_add, labels_to_add = self.pseudo_label_alignment(copy.deepcopy(model),data, buf_data, pseudo_labels, buf_labels, personalized_model, device=self.args.device)

                        # update the buffer
                        #buffer.add_data(examples=data_cluster, labels=pseudo_labels_cluster)
                        buffer.add_data(examples=data_to_add, labels=labels_to_add)
                
                    model.train()
                    for steps in range(self.args.update_step):
                        buf_data, buf_labels = buffer.get_data(self.args.buffer_batch_size, transform=None)        
                        
                        data_comb = torch.cat([data_cluster, buf_data], dim=0)
                        label_comb = torch.cat([pseudo_labels_cluster, buf_labels], dim=0)

                        # shuffle the data
                        indices = torch.randperm(data_comb.size(0))
                        data_comb = data_comb[indices]
                        label_comb = label_comb[indices]

                        optimizer.zero_grad()
                        #print(data.size())
                        
                        output,_ = model(data_comb)
                        loss = criterion(output, label_comb.long())
                        loss.backward()
                        optimizer.step()
                        #print("loss", loss.item())
                    train_loss += loss.item()*data_cluster.size(0)

                    model.eval()
                    output,_ = model(data_cluster)
                    # Get predictions
                    _, predicted = torch.max(output, 1)

                    

                    all_labels.extend(pseudo_labels_cluster.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                epoch_loss = train_loss/len(train_loader.dataset)
                epoch_acc = accuracy_score(all_labels, all_predictions)
                scheduler.step()

                # adaptaion accuracy and loss
                #val_loss, val_acc = validate_model(model, val_loader,buffer)

                val_loss, val_acc = self.align_and_evaluate(model, val_loader, buffer, device=self.args.device)

                sub_emb, sub_labels = self.align_and_display(model, val_loader, buffer, device=self.args.device, buffer_display=False)

                buff_emb, buff_labels = self.align_and_display(model, val_loader, buffer, device=self.args.device, buffer_display=True)

                # generalization accuracy and loss
                #test_gen_loss, test_gen_acc = validate_model(model, test_gen_loader,buffer)
                test_gen_loss, test_gen_acc = self.align_and_evaluate(model, test_gen_loader, buffer, device=self.args.device)

                # forgetting mitigation accuracy and loss
                if subject_id > 1:
                    #test_for_loss, test_for_acc = validate_model(model, test_for_loader,buffer)
                    test_for_loss, test_for_acc = self.align_and_evaluate(model, test_for_loader, buffer, device=self.args.device)



                    # print the results
                    print(f'Epoch [{epoch+1}/{self.args.n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, '
                        f'Test Gen Loss: {test_gen_loss:.4f}, Test Gen Accuracy: {test_gen_acc:.4f}, '
                        f'Test For Loss: {test_for_loss:.4f}, Test For Accuracy: {test_for_acc:.4f}')
                
                else:
                    print(f'Epoch [{epoch+1}/{self.args.n_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, '
                        f'Test Gen Loss: {test_gen_loss:.4f}, Test Gen Accuracy: {test_gen_acc:.4f}')
                
                
                self.plot_umap_emb(buff_emb,buff_labels,subject_id, mode = 'buff')
                self.plot_umap_emb(sub_emb,sub_labels,subject_id, mode = 'sub')


            sub_ids.append(subject_id)
            adaptation_accuracy.append(val_acc)
            generalization_accuracy.append(test_gen_acc)
            if subject_id > 1:
                forgetting_mitigation_accuracy.append(test_for_acc)

            # buffer_emb_display.append(buff_emb)
            # buffer_label_display.append(buff_labels)

            # sub_emb_display.append(sub_emb)
            # sub_label_display.append(sub_labels)


        # print('Mean adaptation accuracy:', torch.stack(adaptation_accuracy).mean())
        # print('Mean generalization accuracy:', torch.stack(generalization_accuracy).mean())
        # print('Mean forgetting mitigation accuracy:', torch.stack(forgetting_mitigation_accuracy).mean())

        print('Mean adaptation accuracy:', torch.mean(torch.tensor(adaptation_accuracy)))
        print('Mean generalization accuracy:', torch.mean(torch.tensor(generalization_accuracy)))
        print('Mean forgetting mitigation accuracy:', torch.mean(torch.tensor(forgetting_mitigation_accuracy)))



        results = [sub_ids, adaptation_accuracy, generalization_accuracy,forgetting_mitigation_accuracy]
        file_name = 'Results_Source_'+ self.args.source_data + 'to_Target_'+ self.args.dataset +'_Seed_' + str(self.args.seed) + '_'
        #np.savetxt(file_name, results, delimiter=',')
        self.save_data(file_name,results)  


    def data_loader(self,subject_id = 1, batch_size = 32):
        train_data, train_labels, test_data, test_labels = self.deap_loader.load_subject_data(subject_id=subject_id)

        if train_data is None:
            return None, None,None
        
        unique_labels = np.unique(train_labels)

        if len(unique_labels) < 2:
            return None, None,None

        print("train_labels shape is: ", train_labels.shape)

        print("test_labels shape is: ", test_labels.shape)


        train_dataset = MultiLabelDataset(train_data, train_labels)
        val_dataset = MultiLabelDataset(test_data, test_labels)

        
        classwise_sampler = SeqSampler(train_dataset, blend_ratio=0.0,
                                n_concurrent_classes=1,
                                batch_size=batch_size)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=classwise_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # load test data for forgetting mitigation
        if subject_id > 1:
            _,test_for_data,_, test_for_labels = self.deap_loader.load_all_deap_data(num_subjects = subject_id)
            test_for_dataset = MultiLabelDataset(test_for_data, test_for_labels)
            test_for_loader = DataLoader(test_for_dataset, batch_size=batch_size, shuffle=True)
            return train_loader, val_loader, test_for_loader
        else:
            return train_loader, val_loader
    


    def cpc_training(self,model,device,data):

        predictor_net = PredictorNet(input_dim=64, hidden_dim=32)
        predictor_net = predictor_net.to(device=device)    

        optimizer = optim.Adam( list(model.parameters()) + list(predictor_net.parameters()), lr=1e-4, weight_decay=1e-5)

        for step in range(10):

            optimizer.zero_grad()
            output,z = model(data)
            
            z = F.normalize(z, p=2, dim=1)  

            z1, z2 = torch.chunk(z, chunks=2, dim=1)  # Split along temporal dimension

            z_pred = predictor_net(z1)

            # Compute CPC loss
            loss = self.cpc_loss(z_pred, z2)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            #print(f"Step {step + 1}, Loss: {loss.item()}")
        
        return model


    def cpc_loss(self,z_pred, z_true, temperature=0.1):
        # z_pred: (batch_size, embedding_dim)
        # z_true: (batch_size, embedding_dim)
        # Compute similarity
        similarity_matrix = torch.mm(z_pred, z_true.T)  # (batch_size, batch_size)
        similarity_matrix /= temperature

        # Create targets (diagonal is positive pair)
        batch_size = z_pred.size(0)
        targets = torch.arange(batch_size).to(z_pred.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, targets)
        return loss
    

    def pseudo_label_alignment(self, model0, batch_data, buf_data, batch_labels, buf_labels, model, device='cuda'):
        model.eval()  # Set model to evaluation mode
        entropy_th = 0.5
        buf_data = buf_data.to(device)
        buf_labels = buf_labels.to(device)

        # Step 1: Compute centroids for each class in the memory buffer
        unique_labels = torch.unique(buf_labels)
        centroids = {}

        with torch.no_grad():
            # Get embeddings for all data in memory buffer
            _, buf_emb = model(buf_data)
            buf_emb = F.normalize(buf_emb, p=2, dim=1)

            # Calculate centroids (mean embeddings) for each class
            for label in unique_labels:
                class_data = buf_emb[buf_labels == label]
                centroids[label] = class_data.mean(dim=0)

        with torch.no_grad():
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Step 2: Extract embeddings for the validation batch
            _, batch_emb = model(batch_data)
            batch_emb = F.normalize(batch_emb, p=2, dim=1)

            # Step 3: Compute cosine similarity between validation samples and centroids
            sim_matrix = cosine_similarity(batch_emb.cpu().detach().numpy(), 
                                            torch.stack(list(centroids.values())).cpu().detach().numpy())

            # Step 4: For each sample, assign the label of the most similar centroid
            pseudo_labels = []
            for i in range(sim_matrix.shape[0]):
                most_similar_class_idx = np.argmax(sim_matrix[i])  # Find the most similar class centroid
                pseudo_label = unique_labels[most_similar_class_idx]  # Assign the class label
                pseudo_labels.append(pseudo_label)

            pseudo_labels = torch.tensor(pseudo_labels).to(device)

            # Store the aligned data and pseudo labels for later usage
            aligned_data = batch_data
            aligned_labels = pseudo_labels

        # concat the buf data and aligned data
        aligned_data_buf_data = torch.cat([buf_data, aligned_data], dim=0)
        aligned_labels_buf_labels = torch.cat([buf_labels, aligned_labels], dim=0)
        _, outputs = model0(aligned_data_buf_data)

        logits = F.softmax(outputs/ self.args.temperature, dim=1)  
        entropies = -torch.sum(logits * torch.log(logits + 1e-9), dim=1)

        selected_indices = []

        # Step 5: Process each class separately and select the top samples based on entropy
        for label in unique_labels:
            # Get the indices of samples corresponding to this class
            class_indices = (aligned_labels_buf_labels == label).nonzero().squeeze()

            # Get the entropies for this class
            class_entropies = entropies[class_indices]

            # Sort the indices based on entropy values (ascending order)
            sorted_class_indices = class_indices[torch.argsort(class_entropies, descending=False)]

            # Select a balanced number of samples (up to 200 in total)
            max_samples_per_class = 200 // len(unique_labels)  # Adjust based on the number of unique labels
            if len(sorted_class_indices) > max_samples_per_class:
                selected_indices.append(sorted_class_indices[:max_samples_per_class])
            else:
                selected_indices.append(sorted_class_indices)

        # Flatten the list of selected indices and ensure they are balanced
        selected_indices = torch.cat(selected_indices)

        # Select the data and labels corresponding to the selected indices
        data_to_add = aligned_data_buf_data[selected_indices]
        labels_to_add = aligned_labels_buf_labels[selected_indices]

        return aligned_data, aligned_labels, data_to_add, labels_to_add


    # def pseudo_label_alignment(self,model0, batch_data, buf_data, batch_labels, buf_labels, model, device='cuda'):
    #     model.eval()  # Set model to evaluation mode
    #     entropy_th = 0.5
    #     buf_data = buf_data.to(device)
    #     buf_labels = buf_labels.to(device)

    #     # Step 1: Compute centroids for each class in the memory buffer
    #     unique_labels = torch.unique(buf_labels)
    #     centroids = {}

    #     with torch.no_grad():
    #         # Get embeddings for all data in memory buffer
    #         _, buf_emb = model(buf_data)
    #         buf_emb = F.normalize(buf_emb, p=2, dim=1)

    #         # Calculate centroids (mean embeddings) for each class
    #         for label in unique_labels:
    #             class_data = buf_emb[buf_labels == label]
    #             centroids[label] = class_data.mean(dim=0)

    #     with torch.no_grad():
    #         batch_data = batch_data.to(device)
    #         batch_labels = batch_labels.to(device)

    #         # Step 2: Extract embeddings for the validation batch
    #         _, batch_emb = model(batch_data)
    #         batch_emb = F.normalize(batch_emb, p=2, dim=1)

    #         # Step 3: Compute cosine similarity between validation samples and centroids
    #         sim_matrix = cosine_similarity(batch_emb.cpu().detach().numpy(), 
    #                                         torch.stack(list(centroids.values())).cpu().detach().numpy())

    #         # Step 4: For each sample, assign the label of the most similar centroid
    #         pseudo_labels = []
    #         for i in range(sim_matrix.shape[0]):
    #             most_similar_class_idx = np.argmax(sim_matrix[i])  # Find the most similar class centroid
    #             pseudo_label = unique_labels[most_similar_class_idx]  # Assign the class label
    #             pseudo_labels.append(pseudo_label)

    #         pseudo_labels = torch.tensor(pseudo_labels).to(device)

    #         # Store the aligned data and pseudo labels for later usage
    #         aligned_data = batch_data
    #         aligned_labels = pseudo_labels

    #     # concat the buf data and aligned data
    #     aligned_data_buf_data = torch.cat([buf_data, aligned_data], dim=0)
    #     aligned_labels_buf_labels = torch.cat([buf_labels, aligned_labels], dim=0)
    #     _, outputs = model0(aligned_data_buf_data)

    #     logits = F.softmax(outputs, dim=1)  

    #     entropies = -torch.sum(logits * torch.log(logits + 1e-9), dim=1)

    #     # # selected_indices = entropies > entropy_th



    #     sorted_indices = torch.argsort(entropies, descending=False)

    #     # Select top entries if size > max_size
    #     if len(sorted_indices) > 200:
    #         selected_indices = sorted_indices[:200]
    #     else:
    #         selected_indices = sorted_indices
        
    #     data_to_add = aligned_data_buf_data[selected_indices]
    #     labels_to_add = aligned_labels_buf_labels[selected_indices]

    #     return aligned_data, aligned_labels, data_to_add, labels_to_add


    def align_and_evaluate(self,model, val_loader, buffer, device='cuda'):
        model.eval()  # Set model to evaluation mode
        correct = 0
        val_loss = 0.0
        total = 0

        buf_data, buf_labels = buffer.get_all_data()  # Get all data from memory buffer
        buf_data = buf_data.to(device)
        buf_labels = buf_labels.to(device)

        # Step 1: Compute centroids for each class in the memory buffer
        unique_labels = torch.unique(buf_labels)
        centroids = {}

        with torch.no_grad():
            # Get embeddings for all data in memory buffer
            _, buf_emb = model(buf_data)
            buf_emb = F.normalize(buf_emb, p=2, dim=1)

            # Calculate centroids (mean embeddings) for each class
            for label in unique_labels:
                class_data = buf_emb[buf_labels == label]
                centroids[label] = class_data.mean(dim=0)

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)


                unique_labels_val = torch.unique(batch_labels)



                # Step 2: Extract embeddings for the validation batch
                _, batch_emb = model(batch_data)
                batch_emb = F.normalize(batch_emb, p=2, dim=1)

                aligned_data = []
                aligned_labels = []

                for label in unique_labels_val:
                    class_data = batch_emb[batch_labels == label]
                    class_data_samples = batch_data[batch_labels == label]
                    centroids_lcl = class_data.mean(dim=0)

                    # compare centroid lcl with centroids
                    sim_matrix = cosine_similarity(centroids_lcl.cpu().detach().numpy().reshape(1,-1), 
                                            torch.stack(list(centroids.values())).cpu().detach().numpy())
                    
                    most_similar_class_idx = np.argmax(sim_matrix)  # Find the most similar class centroid

                    pseudo_label = unique_labels[most_similar_class_idx]  # Assign the class label

                    pseudo_labels = torch.ones(class_data.size(0), dtype=torch.long,device=device) * pseudo_label

                    aligned_data.append(class_data_samples)
                    aligned_labels.append(pseudo_labels)



                aligned_data = torch.cat(aligned_data, dim=0)
                aligned_labels = torch.cat(aligned_labels, dim=0)

                aligned_labels = aligned_labels.to(device)

                # Forward pass
                outputs,_ = model(aligned_data)
                loss = self.criterion(outputs, aligned_labels.long())
                val_loss += loss.item() * aligned_data.size(0)

                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == aligned_labels.data)
                total += aligned_labels.size(0)

        val_loss /= total
        val_acc = correct.double() / total
        return val_loss, val_acc.item()# val_acc

    def align_and_display(self,model, val_loader, buffer, device='cuda', buffer_display = False):
        model.eval()  # Set model to evaluation mode
        correct = 0
        val_loss = 0.0
        total = 0
        all_embeddings = []
        all_labels = []

        buf_data, buf_labels = buffer.get_all_data()  # Get all data from memory buffer
        buf_data = buf_data.to(device)
        buf_labels = buf_labels.to(device)

        # Step 1: Compute centroids for each class in the memory buffer
        unique_labels = torch.unique(buf_labels)
        centroids = {}

        with torch.no_grad():
            # Get embeddings for all data in memory buffer
            _, buf_emb = model(buf_data)
            buf_emb = F.normalize(buf_emb, p=2, dim=1)

            # Calculate centroids (mean embeddings) for each class
            for label in unique_labels:
                class_data = buf_emb[buf_labels == label]
                centroids[label] = class_data.mean(dim=0)

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                # Step 2: Extract embeddings for the validation batch
                _, batch_emb = model(batch_data)
                batch_emb = F.normalize(batch_emb, p=2, dim=1)

                # Step 3: Compute cosine similarity between validation samples and centroids
                sim_matrix = cosine_similarity(batch_emb.cpu().detach().numpy(), 
                                            torch.stack(list(centroids.values())).cpu().detach().numpy())

                # Step 4: For each sample, assign the label of the most similar centroid
                pseudo_labels = []
                for i in range(sim_matrix.shape[0]):
                    most_similar_class_idx = np.argmax(sim_matrix[i])  # Find the most similar class centroid
                    pseudo_label = unique_labels[most_similar_class_idx]  # Assign the class label
                    pseudo_labels.append(pseudo_label)

                pseudo_labels = torch.tensor(pseudo_labels).to(device)

                # Store the aligned data and pseudo labels for later usage
                aligned_data = batch_data
                aligned_labels = pseudo_labels


                # Forward pass
                _,embeddings = model(aligned_data)
                # normalize the embeddings
                all_embeddings.append(embeddings.cpu().detach().numpy())
                all_labels.append(aligned_labels.cpu().detach().numpy())


        if buffer_display:
            return buf_emb.cpu().detach().numpy(), buf_labels.cpu().detach().numpy()
        else:
            return all_embeddings, all_labels



    def plot_umap_emb(self,emb,labels,sub_id, mode = 'buff'):

        emotion = ['LVHA','LVLA','HVHA','HVLA']

        # Set plot style
        plt.style.use('ggplot')

        # Ensure embeddings and labels are properly formatted
        embeddings = np.vstack(emb)  # Stack embeddings into a 2D array (samples x features)
        labels = np.hstack(labels)  # Flatten the labels into a 1D array

        # # Ensure embeddings and labels are properly formatted
        # embeddings = np.vstack(embeddings)  # Stack embeddings into a 2D array (samples x features)
        # labels = np.hstack(labels)  # Flatten the labels into a 1D array

        # Initialize UMAP and fit the data
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_embeddings = umap_model.fit_transform(embeddings)  # Reduce to 2D

        # Use the 'tab10' colormap for distinct colors
        palette = sns.color_palette("tab10", len(np.unique(labels)))

        # Plot UMAP projections
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            x=umap_embeddings[:, 0], y=umap_embeddings[:, 1],
            hue=labels, palette=palette,  # Use the 'tab10' palette
            s=50, alpha=0.7, edgecolor=None
        )

        # Add a horizontal legend at the top
        unique_labels = np.unique(labels)
        colors = sns.color_palette("tab10", len(unique_labels))  # Match the scatterplot colors
        # legend_handles = [
        #     plt.Line2D([0], [0], marker='o', color='black', label=f" {int(label)}",
        #             markerfacecolor=color, markersize=10)
        #     for label, color in zip(unique_labels, colors)
        # ]

        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='black', label=emotion[int(label)],
                    markerfacecolor=color, markersize=10)
            for label, color in zip(unique_labels, colors)
        ]
        plt.legend(
            handles=legend_handles,
            loc='upper center',  # Position the legend at the top center
            bbox_to_anchor=(0.5, 1.2),  # Adjust position (above the plot)
            ncol=min(len(unique_labels), 5),  # Arrange in up to 5 columns
            fontsize=15
        )
        # Display the plot
        plt.tight_layout()
        dir_save = r'C:\Users\adnan\Desktop\Final\Second Study\Code\SSCL\umap\\'+ self.args.dataset +'\\source_'
        
        if mode == 'buff':
            dir_save = dir_save + self.args.source_data + '_buff' + str(sub_id) + 'Temp_'+ str(self.args.temperature) + '.png'
        else:
            dir_save = dir_save + self.args.source_data +'_subject_' + str(sub_id) + 'Temp_'+ str(self.args.temperature)  + '.png'
        # save the plot
        plt.savefig(dir_save)
        #plt.show()


    def save_data(self, name,d):
        export_data = zip_longest(*d, fillvalue = '')
        with open(name+'.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(("Subject_ID", "adapt_acc", "Gen_acc", "Forg_acc"))
            wr.writerows(export_data)
        myfile.close()



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