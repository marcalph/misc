import multiprocessing as mp
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data directory
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


# test/train split
img_fns_train, img_fns_val = train_test_split(images, random_state=42)

char2int = dict(zip(["-"]+characters, range(len(characters)+1)))
int2char = dict(zip(char2int.values(), char2int.keys()))


class CaptchaDataset(Dataset):
    def __init__(self, image_fns):
        self.image_fns = image_fns

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_fns[idx]
        image = cv2.imread(str(img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255
        image = self.transform(image)
        target = torch.Tensor([char2int[c] for c in self.image_fns[idx].split(os.path.sep)[-1].split(".png")[0]])
        # print(labels[idx])
        return image, target

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToTensor()
            ])
        return transform_ops(image)


train_ds = CaptchaDataset(img_fns_train)
train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_ds = CaptchaDataset(img_fns_val)
val_dataloader = DataLoader(val_ds, batch_size=16)#, shuffle=True)


for imgs, lbls in train_dataloader:
    print(imgs.size())
    print(len(lbls))
    print(lbls)
    break


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
            nn.Conv2d(1, 32, 3, padding=1),
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
        x = x.permute(0, 3, 1, 2)
        # print("permute", x.size())
        x = x.reshape((x.size(0), 50, -1))
        # print("reshape", x.size())
        x = self.projection(x)
        # print("project", x.size())
        x, _ = self.rnn1(x) 
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


model = OCRNet()
model.apply(weights_init)
model = model.to(device)
crit = nn.CTCLoss()
opt = optim.AdamW(model.parameters())


def train(model, data, opt, crit):
    model.train()
    epoch_losses = []
    for i, (inputs, targets) in enumerate(tqdm(data, leave=False)):
        opt.zero_grad()
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        targets = targets.view(-1)
        outputs_lens = torch.full(size=(inputs.size(0),), fill_value=outputs.size(0), dtype=torch.long)
        targets_lens = torch.full(size=(inputs.size(0),), fill_value=max_length, dtype=torch.long)
        loss = crit(outputs, targets, outputs_lens, targets_lens)
        epoch_losses.append(loss.item())
        loss.backward()
        opt.step()
    trn_losses.append(np.mean(epoch_losses))


def validate(model, data, opt, crit):
    model.eval()
    epoch_losses = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(data, leave=False)):
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            targets = targets.view(-1)
            outputs_lens = torch.full(size=(inputs.size(0),), fill_value=outputs.size(0), dtype=torch.long)
            targets_lens = torch.full(size=(inputs.size(0),), fill_value=max_length, dtype=torch.long)
            loss = crit(outputs, targets, outputs_lens, targets_lens)
            epoch_losses.append(loss.item())
        val_losses.append(np.mean(epoch_losses))

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/ocr_1')
# get some random training images
dataiter = iter(train_dataloader)
images, labels = dataiter.next()
import torchvision


# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))


# # create grid of images
# img_grid = torchvision.utils.make_grid(images)

# # show images
# matplotlib_imshow(img_grid, one_channel=True)

# # write to tensorboard
# writer.add_image('four_fashion_mnist_images', img_grid)












# trn_losses = []
# val_losses = []
# for epoch in range(10):
#     train(model, train_dataloader, opt, crit)
#     validate(model, val_dataloader, opt, crit)
#     print("epoch:{}     trn loss:{}      val loss:{}".format(epoch, trn_losses[-1], val_losses[-1]))





# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# ax1.plot(trn_losses)
# ax1.set_xlabel("Epochs")
# ax1.set_ylabel("Loss")

# ax2.plot(val_losses)
# ax2.set_xlabel("Epochs")
# ax2.set_ylabel("Loss")

# plt.show()


# def decode(outputs):
#     """ decode predictions with greedy startegy
#     """
#     outputs = outputs.argmax(2)  # [t, batch_size]
#     outputs = outputs.numpy().T
#     text_batch = []
#     for output in outputs:
#         text = "".join([int2char[i] for i in output])
#         text_batch.append(text)

#     return text_batch


# results_train = pd.DataFrame(columns=['actual', 'prediction'])
# with torch.no_grad():
#     for image_batch, text_batch in tqdm(train_dataloader, leave=True):
#         outputs = model(image_batch.float().to(device)) # [T, batch_size, num_classes==num_features]
#         text_batch_pred = decode(outputs.cpu())
#         #print(text_batch, text_batch_pred)
#         df = pd.DataFrame(columns=['actual', 'prediction'])
#         df['actual'] = text_batch
#         df['prediction'] = text_batch_pred
#         results_train = pd.concat([results_train, df])
# results_train = results_train.reset_index(drop=True)


# results_test = pd.DataFrame(columns=['actual', 'prediction'])
# with torch.no_grad():
#     for image_batch, text_batch in tqdm(val_dataloader, leave=True):
#         outputs = model(image_batch.float().to(device)) # [T, batch_size, num_classes==num_features]
#         text_batch_pred = decode(outputs.cpu())
#         #print(text_batch, text_batch_pred)
#         df = pd.DataFrame(columns=['actual', 'prediction'])
#         df['actual'] = text_batch
#         df['test'] = df.actual.apply(lambda x: [int2char[xi.item()] for xi in x])
#         df['prediction'] = text_batch_pred
#         results_test = pd.concat([results_test, df])
# results_test = results_test.reset_index(drop=True)


# print(results_train.shape)
# print(results_train.head())
# print(results_test.shape)
# print(results_test.head())



# def remove_duplicates(text):
#     if len(text) > 1:
#         letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
#     elif len(text) == 1:
#         letters = [text[0]]
#     else:
#         return ""
#     return "".join(letters)


# def correct_prediction(word):
#     parts = word.split("-")
#     parts = [remove_duplicates(part) for part in parts]
#     corrected_word = "".join(parts)
#     return corrected_word


# results_train['prediction_corrected'] = results_train['prediction'].apply(correct_prediction)
# results_train.head()
# results_test['prediction_corrected'] = results_test['prediction'].apply(correct_prediction)
# results_test.sample(10).head()



# mistakes_df = results_test[results_test['actual'] != results_test['prediction_corrected']]
# print(mistakes_df.head())
# print(mistakes_df['prediction_corrected'].str.len().value_counts())
# mask = mistakes_df['prediction_corrected'].str.len() == 5
# print(mistakes_df[mask])
# mistake_image_fp = os.path.join(data_dir, mistakes_df[mask]['actual'].values[0] + ".png")
# print(mistake_image_fp)
# mistake_image = Image.open(mistake_image_fp)
# plt.imshow(mistake_image)
# plt.show()
# train_accuracy = accuracy_score(results_train['actual'], results_train['prediction_corrected'])
# print(train_accuracy)
# test_accuracy = accuracy_score(results_test['actual'], results_test['prediction_corrected'])
# print(test_accuracy)