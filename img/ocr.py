import os
from collections import Counter
from pathlib import Path
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18

import string
from tqdm.notebook import tqdm
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms



global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return "".join(characters).find(char)+1


class CaptchaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data[idx]
        # image = cv2.imread(str(img_name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        # if self.transform:
        #     augmented = self.transform(image=image)
        #     image = augmented['image']
        label = torch.Tensor([char2int(c) for c in self.labels[idx]])
        # print(labels[idx])
        return image, label

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        return transform_ops(image)


train_ds = CaptchaDataset(x_train, y_train)
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_ds = CaptchaDataset(x_valid, y_valid)
valid_dataloader = DataLoader(valid_ds, batch_size=16)#, shuffle=True)


# # visualize the data
# _, ax = plt.subplots(4, 4, figsize=(10, 5))
# for images, labels in train_dataloader:
#     # images = batch["image"]
#     # labels = batch["label"]
#     for i in range(16):
#         img = (images[i].permute(1,2,0)*255).numpy().astype("uint8")
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
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.projection = nn.Sequential(
            nn.Linear(768, 64),
            nn.Dropout(p=0.2)
        )
        self.rnn1 = nn.GRU(64, 128, bidirectional=True, batch_first=True, dropout=0.25)
        self.rnn2 = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.25)
        self.project = nn.Linear(128, len(characters)+1)

    def forward(self, x):
        # print("input batch", x.size())
        x = self.conv_base(x)
        # print("conv ouput", x.size())
        x = x.permute(0,3,1,2)
        # print("permute", x.size())
        x = x.reshape((x.size(0), 50, -1))
        # print("reshape", x.size())
        x = self.projection(x)
        # print("project", x.size())
        x, _ = self.rnn1(x)
        # print("rnn1", x.size())
        x, _ = self.rnn2(x)
        # print("rnn2", x.size())
        x = self.project(x)
        # print("project", x.size())
        x = x.permute(1, 0, 2)
        # print("permute", x.size())
        x = F.log_softmax(x, dim=2)
        # print("logprobs", x.size())
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


from tqdm import tqdm
import torch.optim as optim

model = OCRNet()
model.apply(weights_init)
model = model.to(device)
crit = nn.CTCLoss()
opt = optim.Adam(model.parameters())


epoch_losses = []
iteration_losses = []
num_updates_epochs = []
for epoch in range(50):
    epoch_loss_list = [] 
    num_updates_epoch = 0
    for input, target in train_dataloader:
        opt.zero_grad()
        input, target = input.to(device), target.to(device)
        output = model(input.float())
        target = target.view(-1)
        output_lengths = torch.full(size=(input.size(0),), fill_value=output.size(0), dtype=torch.long)
        target_lengths = torch.full(size=(input.size(0),), fill_value=max_length, dtype=torch.long)
        loss = crit(output, target, output_lengths, target_lengths)
        iteration_loss = loss.item()
        iteration_losses.append(iteration_loss)
        epoch_loss_list.append(iteration_loss)
        loss.backward()
        opt.step()
    epoch_loss = np.mean(epoch_loss_list)
    print("Epoch:{}    Loss:{}".format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(epoch_losses)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

ax2.plot(iteration_losses)
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Loss")

plt.show()

# output_lengths = torch.full(size=(batch_size,), fill_value=max_length, dtype=torch.long)
# print("hello")
# print(output_lengths.size())
# print(images.size())
# print(labels.size())
