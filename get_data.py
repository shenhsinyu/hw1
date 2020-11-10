import pandas as pd
import shutil
import numpy as np
import os

# read data
df = pd.read_csv("/home/sidney/Desktop/hw1/training_labels.csv")
df = df.sort_values(by=['id'])
label = df['label'].tolist()
index = df.index.to_numpy()
file = df["id"].to_numpy()

path = "/home/sidney/Desktop/hw1/training_data"

name = os.listdir(path)
name.sort(key=lambda x: int(x[:-4]))

for i in range(0, 11185):
    if not os.path.exists("/home/sidney/Desktop/hw1/train/"+label[i]):
        os.makedirs("/home/sidney/Desktop/hw1/train/"+label[i])
for i in range(0, 11185):
    old = path + '/' + name[i]
    new = "/home/sidney/Desktop/hw1/train/" + label[i] + '/' + name[i]
    shutil.move(old, new)

