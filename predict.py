from preprocessing_data import AudioUtil
import torch
from model import AudioClassifier
from torch import nn
import pandas as pd
from pathlib import Path
def get_class_id(filepath):
    path = "UrbanSound8K"
    data_path = path + "/metadata/UrbanSound8K.csv"
    # Read metadata file

    df = pd.read_csv(Path(data_path))

    # Construct file path by concatenating fold and file name
    df["relative_path"] = (
        "/fold" + df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
    )
    class_id = df.loc[df["slice_file_name"] == filepath]["classID"].values[0]
    return class_id
def predict_class(model, filepath):
    # Load the saved model
    model.eval()

    # Preprocess the input WAV file

    aud = AudioUtil.open(filepath)
    reaud = AudioUtil.resample(aud, 44100)
    rechan = AudioUtil.rechannel(reaud, 2)

    dur_aud = AudioUtil.pad_trunc(rechan, 4000)
    shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    #Convert the spectrogram to a PyTorch tensor and add batch dimension
    input_tensor = torch.tensor(aug_sgram, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    # Perform inference
    with torch.no_grad():
       output = model(input_tensor)

    # Get predicted class
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

# Load the model
model = nn.DataParallel(AudioClassifier())
model.load_state_dict(torch.load("model.pt"))

# # Path to the WAV file for prediction
file_path = ["7061-6-0-0.wav","14386-9-0-6.wav","14386-9-0-11.wav"
             ,"14386-9-0-16.wav","14386-9-0-17.wav","16860-9-0-26.wav",
             "16860-9-0-30.wav","16860-9-0-50.wav","16860-9-0-45.wav"]

# Predict the class of the WAV file
for i,file in enumerate(file_path): # file_path:  
    predicted_class = predict_class(model, file)
    print("Predicted class of case {}: {}".format(i, predicted_class))
    print("Truth class of case {} is: {}".format(i,get_class_id(file)))
    print("############################################################")