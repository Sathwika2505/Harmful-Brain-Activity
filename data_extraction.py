import pandas as pd
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

