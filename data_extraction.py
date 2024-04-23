import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
import pywt
print("The wavelet functions we can use:")
print(pywt.wavelist())
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import glob


# Load the CSV file
df = pd.read_csv("/home/ubuntu/Object_detection_FCOS/hms-harmful/train.csv") 
print(df.eeg_id.unique())

#replacement_dict = {
#    'Seizure': 0,
#    'LPD': 1,
#    'GPD': 2,
#    'LRDA': 3,
#    'GRDA': 4,
#    'Other': 5
#}
#
## Replace values in the 'expert_consensus' column using .replace() method
#df['expert_consensus'] = df['expert_consensus'].replace(replacement_dict)
#
#print("sdfghjk------------------------", df['expert_consensus'])
#
## Select only the 'expert_consensus' column as the target variable
#TARGETS = 'expert_consensus'
#
## Print some information for debugging
#print('Train shape:', df.shape)
#print('Target:', TARGETS)
#
## Group by 'eeg_id' and select the first occurrence of 'spectrogram_id' and minimum 'spectrogram_label_offset_seconds'
#train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
#    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
#train.columns = ['spec_id','min']
#
#tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
#    {'spectrogram_label_offset_seconds':'max'})
#train['max'] = tmp
#
#tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
#train['patient_id'] = tmp
#
#train[TARGETS] = df.groupby('eeg_id')[TARGETS].first()
#
## Normalize the target
##train[TARGETS] = train[TARGETS] / train[TARGETS].sum(axis=1, keepdims=True)
#
#
#train = train.reset_index()
#print('Train non-overlapp eeg_id shape:', train.shape )
#train.head()
#
#
#spectrograms = np.load("/home/ubuntu/Object_detection_FCOS/understanding_data/specs.npy",allow_pickle=True).item()
#
#READ_EEG_SPEC_FILES = True
#
#if READ_EEG_SPEC_FILES:
#    all_eegs = {}
#    for i,e in enumerate(train.eeg_id.values):
#        if i%100==0: print(i,', ',end='')
#        x = np.load(f'/home/ubuntu/Object_detection_FCOS/understanding_data/EEG_Spectrograms/{e}.npy')
#        all_eegs[e] = x
#    
#print(all_eegs)
#   
#class EEGDataset(Dataset):
#    def __init__(self, data, specs=spectrograms, eeg_specs=all_eegs, mode='train'):
#        self.data = data
#        self.specs = specs
#        self.eeg_specs = eeg_specs
#        self.mode = mode
#
#    def __len__(self):
#        return len(self.data)
#
#    def __getitem__(self, idx):
#        row = self.data.iloc[idx]
#        X, y = self.__data_generation(row)
#        return torch.tensor(X), torch.tensor(y)  # Ensure y is converted to tensor
#
#    def __data_generation(self, row):
#        X = np.zeros((128, 256, 16), dtype='float32')
#        y = np.zeros(1, dtype='float32')  # Change the size of y to 1
#
#        if self.mode == 'test':
#            r = 0
#        else:
#            r = int((row['min'] + row['max']) // 4)
#
#        for k in range(4):
#            img = self.specs[row['spec_id']][r:r + 300, k * 100:(k + 1) * 100].T
#            img = np.clip(img, np.exp(-4), np.exp(8))
#            img = np.log(img)
#            ep = 1e-6
#            m = np.nanmean(img.flatten())
#            s = np.nanstd(img.flatten())
#            img = (img - m) / (s + ep)
#            img = np.nan_to_num(img, nan=0.0)
#            X[14:-14, :, k] = img[:, 22:-22] / 2.0
#
#        eeg_id = row['eeg_id']
#        img = self.eeg_specs[eeg_id]
#        X[:, :, 4:] = img[:, :, :12]
#
#        if self.mode != 'test':
#            # Assign the target label directly to y
#            y = row[TARGETS]
#
#        return X, y  # Ensure y is returned as a single value 
#
#
#eeg_dataset = EEGDataset(train, spectrograms, all_eegs)
#dataloader = DataLoader(eeg_dataset, batch_size=32, shuffle=False)
#
#num_images_saved = 0 
#
#for i, (x, y) in enumerate(dataloader):
#
#    if num_images_saved >= 25:
#        break
#    
#    plt.figure(figsize=(20, 8))
#    ROWS, COLS = 2, 3
#
#    for j in range(ROWS):
#        for k in range(COLS):
#            plt.subplot(ROWS, COLS, j * COLS + k + 1)
#            t = y[j * COLS + k]
#
#            # Get the index of the target
#            #target_index = torch.argmax(t).item()
#            img = x[j * COLS + k, :, :, 0].cpu().numpy()[::-1, ]
#            mn = img.flatten().min()
#            mx = img.flatten().max()
#            img = (img - mn) / (mx - mn)
#            plt.imshow(img)
#            plt.title(f'EEG = {train.eeg_id.values[i * 32 + j * COLS + k]}\nTarget = {t}', size=12)
#            plt.yticks([])
#            plt.ylabel('Frequencies (Hz)', size=14)
#            plt.xlabel('Time (sec)', size=16)
#            
#            num_images_saved += 1
#            if num_images_saved >= 25:
#                break
#
#    # Save the figure with error handling
#    try:
#        plt.savefig(f"/home/ubuntu/Object_detection_FCOS/hms-harmful/brain activity usecase/image_batch_{i}.png")
#    except Exception as e:
#        print(f"Error occurred while saving image {i}: {e}")
#    finally:
#        plt.close()  # Close the figure to release memory