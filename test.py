import torchaudio
import torch
import pandas as pd

#check torch version and the backend_list of the torch audio
print( torch.version)
print(str(torchaudio.list_audio_backends))
#check cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
    torch.cuda.set_device(0)
else:
    device = torch.device("cpu")
    print("Running on the CPU")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

# load metadata
file_name = "UrbanSound8K/metadata/UrbanSound8K.csv"
df = pd.read_csv(file_name)
df["relative_path"]="/fold"+ df['fold'].astype(str) +"/" + df["slice_file_name"].astype(str) 
print(len(df))
