import pandas as pd
filepath = "7061-6-0-0.wav"
path = "UrbanSound8K"
data_path = path + "/metadata/UrbanSound8K.csv"
# Read metadata file

df = pd.read_csv(data_path)

# Construct file path by concatenating fold and file name
df["relative_path"] = (
    "/fold" + df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
)
print(df.head(10))
class_id = df.loc[df["slice_file_name"] == filepath]["classID"].values

print(class_id)
print(df.head(10))