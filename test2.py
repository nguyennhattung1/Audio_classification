import torch

from torch.utils.data import random_split
from pathlib import Path
import pandas as pd
from dataset import SoundDS


# Assuming train_dl is your DataLoader
path = "UrbanSound8K"
data_path = path + "/metadata/UrbanSound8K.csv"
# Read metadata file

df = pd.read_csv(Path(data_path))

# Construct file path by concatenating fold and file name
df["relative_path"] = (
    "/fold" + df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
)
myds = SoundDS(df, path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])
print("Length of train_ds:", len(train_ds))
# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False)
for batch in train_dl:
    # batch is a tuple or dictionary containing your input data and labels
    # Assuming it's a tuple with (inputs, labels)
    inputs, labels = batch
    
    # Check the shape of inputs and labels
    print("Input shape:", inputs.shape)
    print("Labels shape:", labels.shape)

    # You can break the loop after printing the shape of the first batch
    break