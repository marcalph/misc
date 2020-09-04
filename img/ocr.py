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

# vocabulary = ["-"] + characters
# print(len(vocabulary))
# print(vocabulary)
# idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
# print(idx2char)
# char2idx = {v: k for k, v in idx2char.items()}
# print(char2idx)


# num_chars = len(char2idx)
# print(num_chars)
# rnn_hidden_size = 256

# resnet = resnet18(pretrained=True)




# class CRNN(nn.Module):
#     def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
#         super(CRNN, self).__init__()
#         self.num_chars = num_chars
#         self.rnn_hidden_size = rnn_hidden_size
#         self.dropout = dropout

#         # CNN Part 1
#         resnet_modules = list(resnet.children())[:-3]
#         self.cnn_p1 = nn.Sequential(*resnet_modules)

#         # CNN Part 2
#         self.cnn_p2 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=(3,6), stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )
#         self.linear1 = nn.Linear(1024, 256)
#         # RNN
#         self.rnn1 = nn.GRU(input_size=rnn_hidden_size, 
#                             hidden_size=rnn_hidden_size,
#                             bidirectional=True, 
#                             batch_first=True)
#         self.rnn2 = nn.GRU(input_size=rnn_hidden_size, 
#                             hidden_size=rnn_hidden_size,
#                             bidirectional=True, 
#                             batch_first=True)
#         self.linear2 = nn.Linear(self.rnn_hidden_size*2, num_chars)


#     def forward(self, batch):
#         print("start",batch.size()) 
#         batch = self.cnn_p1(batch)
#         print("cnn_part1",batch.size()) # torch.Size([-1, 256, 4, 13])
#         batch = self.cnn_p2(batch) # [batch_size, channels, height, width]
#         print("cnn_part2",batch.size())# torch.Size([-1, 256, 4, 10])
#         batch = batch.permute(0, 3, 1, 2) # [batch_size, width, channels, height]
#         print("permute to have BWCH",batch.size()) # torch.Size([-1, 10, 256, 4])

#         batch_size = batch.size(0)
#         T = batch.size(1)
#         batch = batch.view(batch_size, T, -1) # [batch_size, T==width, num_features==channels*height]
#         print("reshape", batch.size()) # torch.Size([-1, 10, 1024])
#         batch = self.linear1(batch)
#         print("projection", batch.size()) # torch.Size([-1, 10, 256])

#         batch, hidden = self.rnn1(batch)
#         feature_size = batch.size(2)
#         print("rnn1",batch.size())
#         batch = batch[:, :, :feature_size//2] + batch[:, :, feature_size//2:]
#         print("sum", batch.size()) # torch.Size([-1, 10, 256])

#         batch, hidden = self.rnn2(batch)
#         print("rnn2", batch.size()) # torch.Size([-1, 10, 512])

#         batch = self.linear2(batch)
#         print("project", batch.size()) # torch.Size([-1, 10, 20])

#         batch = batch.permute(1, 0, 2) # [T==10, batch_size, num_classes==num_features]
#         print("permute to TBF",batch.size()) # torch.Size([10, -1, 20])
#         return batch



# def weights_init(m):
#     classname = m.__class__.__name__
#     if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0.01)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

# for image_batch, text_batch in train_dataloader:
#     break

# criterion = nn.CTCLoss(blank=0)

# crnn = CRNN(20)
# crnn.apply(weights_init)
# crnn = crnn.to(device)

# text_batch_logits = crnn(image_batch.to(device))
# print("tb",text_batch)
# print("tblogits",text_batch_logits.shape)


# def encode_text_batch(text_batch):
    
#     text_batch_targets_lens = [len(text) for text in text_batch]
#     text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)
    
#     text_batch_concat = "".join(text_batch)
#     print("tb",text_batch)
#     print("tbc",text_batch_concat)
#     text_batch_targets = [char2idx[c] for c in text_batch_concat]
#     text_batch_targets = torch.IntTensor(text_batch_targets)
    
#     return text_batch_targets, text_batch_targets_lens





# def compute_loss(text_batch, text_batch_logits):
#     """
#     text_batch: list of strings of length equal to batch size
#     text_batch_logits: Tensor of size([T, batch_size, num_classes])
#     """
#     text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]  
#     text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
#                                        fill_value=text_batch_logps.size(0), 
#                                        dtype=torch.int32).to(device) # [batch_size] 
#     print("logits shape", text_batch_logps.shape)
#     print("logits lens",text_batch_logps_lens) 
#     text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
#     print("target",text_batch_targets)
#     print("target lens", text_batch_targets_lens)
#     loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)
#     print("hello")
#     return loss

# compute_loss(text_batch, text_batch_logits)



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
        x, _ = nn.GRU(64, 128, bidirectional=True, batch_first=True, dropout=0.25)(x)
        # print("rnn1", x.size())
        x, _ = nn.GRU(256, 64, bidirectional=True, batch_first=True, dropout=0.25)(x)
        # print("rnn2", x.size())
        x = nn.Linear(128, len(characters)+1)(x)
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


for images, labels in train_dataloader:
    break







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
for epoch in range(10):
    epoch_loss_list = [] 
    num_updates_epoch = 0
    for input, target in train_dataloader:
        opt.zero_grad()
        input, target = input.to(device), target.to(device)
        output = model(input.float())
        target = target.reshape(-1)
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
    



# output_lengths = torch.full(size=(batch_size,), fill_value=max_length, dtype=torch.long)
# print("hello")
# print(output_lengths.size())
# print(images.size())
# print(labels.size())
