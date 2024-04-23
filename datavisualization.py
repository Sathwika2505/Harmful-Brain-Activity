import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import boto3
import io
import pathlib
from PIL import Image
import multiprocessing
from functools import partial
import gc

# Function to load data from S3
def load_data_from_s3(bucket_name, file_key):
    try:
        print("Accessing file from S3")
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        eeg_specs_data = response['Body'].read()
        return np.load(io.BytesIO(eeg_specs_data), allow_pickle=True).item()
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return None

# Function to process and save images
def process_and_save_images(eeg_id, img, train, spectrograms, output_dir):
    label_folders = {
        0: "Seizure",
        1: "LPD",
        2: "GPD",
        3: "LRDA",
        4: "GRDA",
        5: "Other"
    }
    corresponding_value = train.loc[train['eeg_id'] == int(eeg_id), 'expert_consensus'].values[0]
    folder_name = label_folders[corresponding_value]
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(16):
        row_index = i // 4
        col_index = i % 4
        axes[row_index, col_index].imshow(img[:, :, i], aspect='auto', origin='lower')
        axes[row_index, col_index].set_title(f'EEG Channel {i + 1}')
        axes[row_index, col_index].axis('off')
    filename = f"image_{eeg_id}.jpg"
    file_path = os.path.join(folder_path, filename)
    plt.savefig(file_path)
    plt.close()
    return filename

# Main function
def main():
    # Load data from S3
    bucket_name = 'deeplearning-mlops-demo'
    file_key = 'new_eeg_specs.npy'
    spectrograms = load_data_from_s3(bucket_name, file_key)
    if spectrograms is None:
        return

    # Load DataFrame
    df = pd.read_csv("your_dataframe.csv")  # Replace "your_dataframe.csv" with your actual DataFrame file
    replacement_dict = {
        'Seizure': 0,
        'LPD': 1,
        'GPD': 2,
        'LRDA': 3,
        'GRDA': 4,
        'Other': 5 
    }
    df['expert_consensus'] = df['expert_consensus'].replace(replacement_dict)

    # Process images
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    train = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg(
        {'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': ['min', 'max']})
    train.columns = ['spec_id', 'min', 'max']
    tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
    train['patient_id'] = tmp
    train['expert_consensus'] = df.groupby('eeg_id')['expert_consensus'].first()
    train = train.reset_index()

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func = partial(process_and_save_images, train=train, spectrograms=spectrograms, output_dir=output_dir)
        saved_files = pool.starmap(func, spectrograms.items())

    print("Output directory:", output_dir)

if __name__ == "__main__":
    main()
