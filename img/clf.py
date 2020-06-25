import itertools
from pathlib import Path
import matplotlib.pyplot as plt
from fastai.metrics import error_rate
from fastai.vision import *
from PIL import Image


bs = 64

path = untar_data(URLs.PETS, dest="./data")

path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)
print(fnames[:5])





# ============
# faw
# ============
def faw_step1():
    print("=== create data gen")
    np.random.seed(2)
    pat = r'/([^/]+)_\d+.jpg$'
    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)
    print(data.classes)
    print(len(data.classes), data.c)
    print("=== instantiate learner")
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    print(learn)
    print("=== training::stage-1::clr")
    learn.fit_one_cycle(4)
    learn.save('stage-1')
    print("=== evaluation")
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    interp.most_confused(min_val=2)
    return data, learn

def faw_interp(data, learn):
    print("=== results interpretation")
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    len(data.valid_ds)==len(losses)==len(idxs)
    interp.plot_top_losses(9, figsize=(15,11))


def faw_step2(learn):
    learn.unfreeze()
    learn.fit_one_cycle(1)
    learn.load('stage-1')
    learn.recorder.plot()
    learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))




# ============
# perso
# ============


from albumentations import (Rotate,Resize, RandomCrop,
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from skimage import io
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

from albumentations.pytorch import ToTensor

import cv2
from PIL import Image
import time
import copy
from torchvision import models




class PetDataset(Dataset):
    def __init__(self, img_path, table_file, transform=None):
        self.img_path = img_path
        self.df = pd.read_table(table_file, names=["filename", "target"], usecols=[0, 1], sep=" ")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = (self.img_path/self.df.iloc[idx, 0]).with_suffix(".jpg")
        image = cv2.imread(str(img_name))

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = self.df.iloc[idx, 1]

        return image, label


train_transforms = Compose([
        # RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1),
        Resize(256, 256), 
        RandomCrop(224, 224),
        HorizontalFlip(p=0.5),
        OpticalDistortion(p=0.75),
        Rotate(limit=10, interpolation=1, p=0.75),
        RandomBrightnessContrast(p=0.75),
        ToTensor()])

val_transforms = Compose([
        # RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1),
        Resize(256, 256),
        ToTensor()])

train_ds = PetDataset(path_img, path_anno/"trainval.txt", train_transforms)
train_dataloader = DataLoader(train_ds, batch_size=64,
                        shuffle=True, num_workers=4)

test_ds = PetDataset(path_img, path_anno/"test.txt", val_transforms)
test_dataloader = DataLoader(test_ds, batch_size=64,
                        shuffle=True, num_workers=4)






def create_model(num_classes):
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model, input_size


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, input_size = create_model(37)
model.to(device)
print(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


dataloaders_dict = {"train": train_dataloader, "val": test_dataloader}
criterion = nn.CrossEntropyLoss()




def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



import torch
from torch import nn

class CLR():
    def __init__(self, train_dataloader, base_lr=1e-5, max_lr=100):
        self.base_lr = base_lr  # lower boundary for lr (initial lr)
        self.max_lr = max_lr  # upper boundary for lr
        self.bn = len(train_dataloader) - 1  # number of iterations used for this test run
        ratio = self.max_lr/self.base_lr  # n
        self.mult = ratio ** (1/self.bn)  # q = (max_lr/init_lr)^(1/n)
        self.best_loss = 1e9  # our assumed best loss
        self.iteration = 0  # current iteration, initialized to 0
        self.lrs = []
        self.losses = []

    def calc_lr(self, loss):
        self.iteration +=1
        if math.isnan(loss) or loss > 4 * self.best_loss:  # stopping criteria
            return -1
        if loss < self.best_loss and self.iteration > 1:
            self.best_loss = loss
        mult = self.mult ** self.iteration  # q = q^i
        lr = self.base_lr * mult  # lr_i = init_lr * q
        self.lrs.append(lr)
        self.losses.append(loss)
        return lr

    def plot(self, start=10, end=-5): # plot lrs vs losses
        plt.xlabel("Learning Rate")
        plt.ylabel("Losses")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log') # learning rates are in log scale


loss_func = nn.CrossEntropyLoss()  # loss function
opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.95) # optimizer
clr = CLR(train_dataloader) # CLR instance

def find_lr(clr):
    running_loss = 0. 
    avg_beta = 0.98 # useful in calculating smoothed loss
    model.train() # set the model in training mode
    for i, (input, target) in enumerate(train_dataloader):
        input, target = input.to(device), target.to(device) # move the inputs and labels to gpu if available
        output = model(var_ip) # predict output
        loss = loss_func(output, var_tg) # calculate loss 

        # calculate the smoothed loss 
        running_loss = avg_beta * running_loss + (1-avg_beta) *loss # the running loss
        smoothed_loss = running_loss / (1 - avg_beta**(i+1)) # smoothening effect of the loss 

        lr = clr.calc_lr(smoothed_loss) # calculate learning rate using CLR
        if lr == -1 : # the stopping criteria
            break
        for pg in opt.param_groups: # update learning rate
            pg['lr'] = lr

        # compute gradient and do parameter updates
        opt.zero_grad()
        loss.backward()
        opt.step()





model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10)



























def get_random_batch(path, file_ext=None, n=16):
    path = Path(path)
    if file_ext:
        files = path.glob("**/*" + file_ext)
    else:
        files = path.rglob("**/*")
    return list(itertools.islice(files, n))


def batch_plot(img_batch, label_batch=None, side=4, ):
    """
    viz function to inspect image data
    displays `side`*2 images of a batch by default
    """
    plt.figure(figsize=(2 * side, 2 * side))
    for i in range(side**2):
        plt.subplot(side, side, i + 1)
        plt.yticks([])
        plt.xticks([])
        plt.grid(False)
        img = Image.open(img_batch[i])
        plt.imshow(img.resize((224, 224)), cmap=plt.cm.binary)
        if label_batch is not None:
            plt.xlabel(label_batch[i])
    plt.show()
    return None


img_batch = get_random_batch(path, ".jpg")
anno_batch = [p.stem for p in img_batch]
batch_plot(img_batch, anno_batch)

for tfm in get_transforms():
    print(tfm)


def augment():
    return Compose([
        RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1),
        HorizontalFlip(p=0.5),
        OpticalDistortion(p=0.75),
        Rotate(limit=10, interpolation=1, p=0.75),
        RandomBrightnessContrast(p=0.75)])

