from torch.utils.data import random_split, DataLoader
from dataset import SoundDS
from pathlib import Path
import pandas as pd
#from metadata import df, data_path


myds = SoundDS(df, data_path)

# random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])


# create training and validation data loaders
# each batch has 2 tensors ( mel spectrogram and class ID)
# the size of batch is (batch_size, num_chanels, Mel freq_bands, time_steps).

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=True)
