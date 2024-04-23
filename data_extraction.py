import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
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
from io import BytesIO
import boto3
import io

# Initialize the S3 client
s3 = boto3.client('s3')

bucket_name = 'usecases-cleandata'
file_key = 'kaggle-competition-dataset/train.csv'

# Download the file from S3
try:
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    eeg_specs_data = response['Body'].read()

    # Read the downloaded file using Pandas
    df = pd.read_csv(io.BytesIO(eeg_specs_data))

    # Now you can work with the DataFrame 'df'
    print(df)

except Exception as e:
    print(f"Error downloading or reading file from S3: {e}")




#import boto3 
#  
## Creating an S3 access object 
#s3 = boto3.client("s3") 
#s3.upload_file( 
#    Filename="/home/ubuntu/Object_detection_FCOS/hms-harmful/eeg_spectrograms16diff.npy", 
#    Bucket="deeplearning-mlops-demo", 
#    Key="eeg_spectrograms16diff.npy"
#)