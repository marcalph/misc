import os
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Path to the data directory
data_dir = Path("./data/captcha_images_v2/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)
print("Longest captcha: ", max_length)

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 200
img_height = 50

# Factor by which the image is going to be downsampled
# by the convolutional blocks. We will be using two
# convolution blocks and each block will have
# a pooling layer which downsample the features by a factor of 2.
# Hence total downsampling factor would be 4.
downsample_factor = 4



# Test/train split 
def split_data(images, labels, train_size=0.9, shuffle=False):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def char2int(char):
    return "".join(characters).find(char)


class CaptchaDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data[idx]
        image = cv2.imread(str(img_name))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor([char2int(c) for c in self.labels[idx]])
        # print(labels[idx])
        return image, label


train_ds = CaptchaDataset(x_train, y_train)
train_dataloader = DataLoader(train_ds, batch_size=16)#, shuffle=True)


# # visualize the data
# _, ax = plt.subplots(4, 4, figsize=(10, 5))
# for images, labels in train_dataloader:
#     # images = batch["image"]
#     # labels = batch["label"]
#     for i in range(16):
#         img = (images[i]*255).numpy().astype("uint8")
#         label = f"{labels[i]}"
#         ax[i // 4, i % 4].imshow(img, cmap="gray")
#         ax[i // 4, i % 4].set_title(label)
#         ax[i // 4, i % 4].axis("off")
#     break
# plt.show()





class OCRNet(nn.Module):
    def __init__(self,):
        super(OCRNet, self).__init__()
        self.conv_base = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.recurrent_base = nn.Sequential(
            nn.Linear(50, 64),
            nn.Dropout(p=0.2),
            # nn.LSTM(768, 128, bidirectional=True, batch_first=True, dropout=0.25),
            # nn.LSTM(128, 256, bidirectional=True, batch_first=True, dropout=0.25),
        )

    def forward(self, x):
        x = self.conv_base(x)
        x = x.view((-1, img_height//4*64, img_width//4))
        x= self.recurrent_base(x)
        return x


for images, labels in train_dataloader:
    break


print(images.size())
model = OCRNet()
output = model(images.unsqueeze(1).float())
print(output.size())



