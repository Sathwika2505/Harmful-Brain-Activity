from feature_engineering import transform_data
import pickle

def save_artifact():
    model_dataset = transform_data()
    # torch.save(model_dataset, "test.pt")
    # model_dataset = torch.load('test.pt')
    with open('eeg_data.pkl', 'wb') as f:
        pickle.dump(model_dataset, f)
    return model_dataset