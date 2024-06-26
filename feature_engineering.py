import torchvision
import torchvision.transforms as transforms
import os
from torchvision import datasets
import pickle

def transform_data():
    output_dir = os.path.join(os.getcwd(),"images")
    
    data_transform = torchvision.transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    model_dataset = datasets.ImageFolder(output_dir, transform=data_transform)
    img, ann = model_dataset[1]
    print("iiiiiiii",img)
    print("aaaaaaaaa:",ann)
    with open('eeg_data.pkl', 'wb') as f:
        pickle.dump(model_dataset, f)
    return model_dataset

transform_data()

