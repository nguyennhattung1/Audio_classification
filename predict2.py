import torch
from pathlib import Path
import pandas as pd
from preprocessing_data import AudioUtil
from torch import nn
from residual_block import Residual

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
def predict_class(model,filepath):
    # Load the saved model
    model.eval()

    # Preprocess the input WAV file

    aud = AudioUtil.open(filepath)
    reaud = AudioUtil.resample(aud, 44100)
    rechan = AudioUtil.rechannel(reaud, 1)

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
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride = 2, padding = 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
def resnet_block(input_channels, num_channels, num_residuals,
                    first_block = False):
        layers = []
        for i in range (num_residuals):
            if i ==0 and not first_block:
                layers.append(Residual(num_channels,use_1x1conv=True, strides=2))
            else:
                layers.append(Residual(num_channels))
        return layers
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
net.load_state_dict(torch.load("net.pt"))

# # Path to the WAV file for prediction
file_path = ["7061-6-0-0.wav","14386-9-0-6.wav","14386-9-0-11.wav"
             ,"14386-9-0-16.wav","14386-9-0-17.wav","16860-9-0-26.wav",
             "16860-9-0-30.wav","16860-9-0-50.wav","16860-9-0-45.wav"]

# Predict the class of the WAV file
for i,file in enumerate(file_path): # file_path:  
    predicted_class = predict_class(net, file)
    print("Predicted class of case {}: {}".format(i, predicted_class))
    print("Truth class of case {} is: {}".format(i,get_class_id(file)))
    print("############################################################")