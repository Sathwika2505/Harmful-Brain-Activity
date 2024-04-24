import matplotlib.pyplot as plt
import boto3
import io
import pathlib
from PIL import Image
import random
from data_extraction import df
import numpy as np, os


replacement_dict = {
    'Seizure': 0,
    'LPD': 1,
    'GPD': 2,
    'LRDA': 3,
    'GRDA': 4,
    'Other': 5
}

df['expert_consensus'] = df['expert_consensus'].replace(replacement_dict)

print("sdfghjk------------------------", df['expert_consensus'])

# Select only the 'expert_consensus' column as the target variable
TARGETS = 'expert_consensus'

# Print some information for debugging
print('Train shape:', df.shape)
print('Target:', TARGETS)

train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
train.columns = ['spec_id','min']

tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
    {'spectrogram_label_offset_seconds':'max'})
train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

train[TARGETS] = df.groupby('eeg_id')[TARGETS].first()

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape )
train.head()
print("train eeg====================",train['eeg_id'])
#train.to_csv("train_final.csv",index=False)
access_key = os.environ.get("AWS_ACCESS_KEY_ID")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
#bucket_name = os.environ.get("Bucket_Name")
bucket_name = 'deeplearning-mlops'
file_key = 'eeg_spectrograms16diff.npy'

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

## Function to load data from S3
def load_data_from_s3(bucket_name, file_key):
    try:
        print("Accessing file from S3")
        print("bucket:" , bucket_name)
        s3 = boto3.client('s3', aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      region_name='us-east-1')
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        eeg_specs_data = response['Body'].read()
        spectrograms = np.load(io.BytesIO(eeg_specs_data), allow_pickle=True).item()
        print("------------:",spectrograms)
        return spectrograms
        
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        print(f"Error loading numpy array from binary data: {e}")
        return None
        


spectrograms = load_data_from_s3(bucket_name, file_key)
print("File loaded")

def save_eeg_images(spectrograms, train, replacement_dict):
    saved_files = []

    # Create folders for each label if they don't exist
    for label in replacement_dict.keys():
        folder_path = f"images/{label}"
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    # Iterate over each EEG data
    for eeg_id, eeg_data in spectrograms.items():
        # Get label for the current EEG data
        label = train[train['eeg_id'] == eeg_id]['expert_consensus'].values[0]
        label_name = [key for key, value in replacement_dict.items() if value == label][0]

        # Create image from EEG data
        plt.figure(figsize=(10, 5))
        for i in range(eeg_data.shape[1]):
            plt.subplot(4, 4, i + 1)
            plt.plot(range(10_000), eeg_data[:, i])
            plt.title(f'channel {i}')
            plt.axis('off')

        # Save image to respective folder
        image_path = f"images/{label_name}/{eeg_id}.jpg"
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

        saved_files.append(image_path)

    print("Images saved successfully.")
    return saved_files
    
saved_files = save_eeg_images(spectrograms, train, replacement_dict)
print("Saved Files:", saved_files)

def open_random_image(path):
        # Get a list of all files in the folder
        all_files = os.listdir(path)
        random_image_file = random.choice(all_files)
        image_path = os.path.join(path, random_image_file)
        image = Image.open(image_path)
        return image
GPA = open_random_image(os.path.join(os.getcwd(),"images/GPA"))
GRDA = open_random_image(os.path.join(os.getcwd(),"images/GRDA"))
LPD = open_random_image(os.path.join(os.getcwd(),"images/LPD"))
Seizure = open_random_image(os.path.join(os.getcwd(),"images/Seizure"))
LRDA = open_random_image(os.path.join(os.getcwd(),"images/LRDA"))
Other = open_random_image(os.path.join(os.getcwd(),"images/Other"))
GPA.save('GPA.jpg')
GRDA.save('GRDA.jpg')
LPD.save('LPD.jpg')
Seizure.save('Seizure.jpg')
LRDA.save('LRDA.jpg')
Other.save('Other.jpg')
