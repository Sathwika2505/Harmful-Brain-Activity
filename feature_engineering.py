from datavisualization import save_and_display_images, output_dir
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision.datasets import ImageFolder
from torchvision import datasets
import pickle

def transform_data():
    #path,Seizure,LPD,GPD,LRDA,GRDA,Other = process_and_save_images()
    saved_files = save_and_display_images()
    output_dir = output_dir
    #print(saved_files)
    #output_dir = "/home/ubuntu/Object_detection_FCOS/hms-harmful/brain activity usecase/Training/"
    data_transform = torchvision.transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #total_list = Seizure + LPD + GPD + LRDA + GRDA + Other
    model_dataset = datasets.ImageFolder(output_dir, transform=data_transform)
    img, ann = model_dataset[1]
    print("iiiiiiii",img)
    print("aaaaaaaaa:",ann)
    with open('eeg_data.pkl', 'wb') as f:
        pickle.dump(model_dataset, f)
    return model_dataset

transform_data()

