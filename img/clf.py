import math
import pathlib
import time
# int(len(dataset) * num_epochs /bs)
from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations import (CLAHE, Blur, Compose, Flip, GaussNoise,RandomSizedCrop,
                            GridDistortion, HorizontalFlip, HueSaturationValue,
                            IAAAdditiveGaussianNoise, IAAEmboss,
                            IAAPerspective, IAAPiecewiseAffine, IAASharpen,
                            MedianBlur, MotionBlur, OneOf, OpticalDistortion,
                            RandomBrightnessContrast, RandomContrast,RandomBrightness,
                            RandomRotate90, Resize, Rotate, ShiftScaleRotate,RandomGamma,NoOp,
                            Transpose)

import albumentations as A
from albumentations.pytorch import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.notebook import tqdm
from fastai.vision import *
from sklearn.model_selection import train_test_split




# todo use lr scheduler from torch
# todo correct label mapping

bs = 64

path = untar_data(URLs.PETS, dest="./data")
path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)
print(fnames[:5])
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print(device)


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
    return data, learn


def faw_interp(data, learn):
    print("=== evaluation")
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    interp.most_confused(min_val=2)
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


def update_lr(optimizer, lr,mom):
    for pg in optimizer.param_groups:
        pg['lr'] = lr
        b1, b2 = pg["betas"]
        pg['betas'] = mom, b2

def accuracy(output, target, is_test=False):
    global total
    global correct
    batch_size = output.shape[0]
    total += batch_size
    _, pred = torch.max(output, 1)
    if is_test:
        preds.extend(pred)
    correct += (pred == target).sum()
    return 100 * correct / total


df = pd.concat([pd.read_table(path_anno/"trainval.txt", names=["filename", "target"], usecols=[0, 1], sep=" "), 
               pd.read_table(path_anno/"test.txt", names=["filename", "target"], usecols=[0, 1], sep=" ")])
df["target"] = df.target.apply(lambda x:x-1)

train_df, val_df = train_test_split(df, test_size=1478)

class PetDataset(Dataset):
    def __init__(self, img_path, df, transform=None):
        self.img_path = img_path
        self.df = df
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
        Resize(256, 256),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT),
            A.NoOp()
        ], p=1),
        A.RandomSizedCrop(min_max_height=(196,256),
                          height=224,
                          width=224, p=1),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5,
                                       contrast_limit=0.4),
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ], p=1),
        A.OneOf([
            A.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=5),
            A.NoOp()
        ], p=1),
        A.OneOf([
            A.CLAHE(),
            A.NoOp()
        ], p=1),
        A.HorizontalFlip(p=0.5),
        ToTensor()
    ])

val_transforms = Compose([
        # RandomResizedCrop(height=224, width=224, scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1),
        Resize(256, 256),
        ToTensor()])

train_ds = PetDataset(path_img, train_df, train_transforms)
train_dataloader = DataLoader(train_ds, batch_size=64,
                        shuffle=True, num_workers=4)
val_ds = PetDataset(path_img, val_df, train_transforms)
val_dataloader = DataLoader(val_ds, batch_size=64,
                        shuffle=True, num_workers=4)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}




class LRFinder():
    """ LRFinder class
        performs lr range test
    """
    def __init__(self, base_lr, max_lr, dataloader):
        self.base_lr = base_lr                                  # lower boundary for lr (initial lr)
        self.max_lr = max_lr                                    # upper boundary for lr
        self.dataloader = dataloader
        self.n = len(dataloader) - 1                            # number of iterations
        self.q = (self.max_lr/self.base_lr) ** (1/self.n)       # q = (max_lr/init_lr)^(1/n)
        self.best_loss = math.inf
        self.lrs = []
        self.losses = []

    def compute_lr_step(self, loss, iteration):
        # stopping criteria
        if math.isnan(loss) or loss > 4 * self.best_loss:
            return -1
        if loss < self.best_loss and iteration > 1:
            self.best_loss = loss
        lr = self.base_lr * self.q ** iteration                 # lr_i = init_lr * q^i
        self.lrs.append(lr)
        self.losses.append(loss)
        return lr

    def search(self, model, optimizer, loss_func, beta=0):
        running_loss = 0.
        beta = 0.                                              # smoothing param
        model.train()                                               # set the model in training mode

        for i, (input, target) in enumerate(tqdm(self.dataloader)):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_func(output, target)

            # smoothing loss by exponential moving average
            running_loss = beta*running_loss + (1 - beta)*loss
            smoothed_loss = running_loss/(1 - beta**(i+1))

            # change lr
            lr = self.compute_lr_step(smoothed_loss, i)
            if lr == -1:
                break
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # compute gradient and update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

    def plot_search(self, start=0, end=-1):
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.plot(self.lrs[start:end], self.losses[start:end])
        plt.xscale('log')                                           # learning rates are in log scale

    def plot_lr_sched(self):
        plt.ylabel("Learning Rate")
        plt.xlabel("Iter")
        plt.plot(range(len(self.lrs)), self.lrs)



def accuracy(output, target, is_test=False):
    global total
    global correct
    batch_size = output.size(0)
    total += batch_size

    _, pred = torch.max(output.data, 1)
    if is_test:
        preds.extend(pred)
    correct += (pred == target).sum().item()
    return 100 * correct / total


class AvgStats(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses =[]
        self.precs =[]
        self.its = []

    def append(self, loss, prec, it):
        self.losses.append(loss)
        self.precs.append(prec)
        self.its.append(it)

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
  
def create_model(num_classes, train_bn=True):
    model = models.resnet34(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    if train_bn:
      for mod in model.modules():
          if isinstance(mod, nn.BatchNorm1d) or isinstance(mod, nn.BatchNorm2d):
              for prm in mod.parameters():
                  prm.requires_grad=True
              
    num_features = model.fc.in_features
    model.avgpool = nn.Sequential(
        # nn.AdaptiveAvgPool2d(1),
        # nn.AdaptiveMaxPool2d(1)
        AdaptiveConcatPool2d(1)
        )
    fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes))
    model.fc = fc_layers
    return model





class OneCyclePolicy():
    def __init__(self, cycle_len, step_per_epoch, max_lr, momentum_vals=(0.95, 0.85), up_phase=0.3, lr_fold=25):
        self.step_per_epoch = step_per_epoch
        self.lr_fold = lr_fold
        self.max_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.up_phase = up_phase
        self.iteration = 0
        self.lrs = []
        self.moms = []
        self.trn_accs = []
        self.val_accs = []
        self.trn_losses = []
        self.val_losses = []
        self.cycle_len = cycle_len
        self.lr_generator = self.lr_gen()
        self.mom_generator = self.mom_gen()

    def lr_gen(self):
        up_prcnt = self.up_phase
        # annihilation_phase = int(self.end_phase*self.step_per_epoch*self.cycle_len)
        up_phase = int(up_prcnt*self.step_per_epoch*self.cycle_len)
        down_phase = self.step_per_epoch*self.cycle_len-up_phase  # -annihilation_phase
        return cycle(np.hstack([np.linspace(self.max_lr/self.lr_fold, self.max_lr, up_phase),  # increase
                                np.linspace(self.max_lr, self.max_lr/self.lr_fold/1000, down_phase)]))  # decrease
                                # np.linspace(self.max_lr/self.lr_fold, self.max_lr/self.lr_fold/10000, annihilation_phase)]))

    def mom_gen(self):
        down_prcnt = self.up_phase
        # const_phase = int(self.end_phase*self.step_per_epoch*self.cycle_len)
        down_phase = int(down_prcnt*self.step_per_epoch*self.cycle_len)
        up_phase = self.step_per_epoch*self.cycle_len-down_phase
        return cycle(np.hstack([np.linspace(self.high_mom, self.low_mom, down_phase),  # decrease
                                np.linspace(self.low_mom, self.high_mom, up_phase)]))  # increase


    def recompute(self):
        self.iteration += 1
        lr = self.compute_lr()
        mom = self.compute_mom()
        return (lr, mom)

    def compute_lr(self):
        lr = next(self.lr_generator)
        self.lrs.append(lr)
        return lr

    def compute_mom(self):
        mom = next(self.mom_generator)
        self.moms.append(mom)
        return mom

    def train(self, model, dataloader, loss_func, optimizer, one_cycle=True):
        model.train()
        running_loss = 0.
        running_corrects = 0
        avg_beta = 0.98
        for i, (input, target) in enumerate(tqdm(dataloader, leave=False)):
            input, target = input.to(device), target.to(device)
            if one_cycle:
                lr, mom = self.recompute()
                update_lr(optimizer, lr,mom)
                # update_mom(learn.opt, mom)

            output = model(input)
            _, preds = torch.max(output, 1)
            loss = loss_func(output, target)

            # running_loss = avg_beta * running_loss + (1-avg_beta) * loss
            # smoothed_loss = running_loss / (1 - avg_beta**(i+1))

            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == target.data)

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        self.trn_losses.append(running_loss / len(dataloader.dataset))
        self.trn_accs.append(running_corrects.item() / len(dataloader.dataset))

    def validate(self, model, dataloader, loss_func, optimizer):
        model.eval()
        running_loss = 0.
        running_corrects = 0
        avg_beta = 0.98
        with torch.no_grad():
        # if train_bn:
            for i, (input, target) in enumerate(tqdm(dataloader, leave=False)):
                input, target = input.to(device), target.to(device)
                output = model(input)
                _, preds = torch.max(output, 1)
                loss = loss_func(output, target)

                # running_loss = avg_beta * running_loss + (1-avg_beta) * loss
                # smoothed_loss = running_loss / (1 - avg_beta**(i+1))

                running_loss += loss.item() * input.size(0)
                running_corrects += torch.sum(preds == target.data)

            # measure accuracy and record loss
            self.val_losses.append(running_loss / len(dataloader.dataset))
            self.val_accs.append(running_corrects.item() / len(dataloader.dataset))

    def fit(self, epochs, model, dataloaders, loss_func, optimizer, one_cycle=True):
        print(f"{'epoch':5s}{'train_loss':>15s}{'valid_loss':>15s}{'train_acc':>15s}{'valid_acc':>15s}")
        for epoch in tqdm(range(epochs), leave=False):
            self.train(model, dataloaders["train"], loss_func, optimizer, one_cycle)
            self.validate(model, dataloaders["val"], loss_func, optimizer)
            print(f"{epoch+1:5}{self.trn_losses[-1]:15.5f}{self.val_losses[-1]:15.5f}{self.trn_accs[-1]:15.5f}{self.val_accs[-1]:15.5f}")
    
    def plot_lr_sched(self):
        plt.ylabel("Learning Rate")
        plt.xlabel("Iter")
        plt.plot(range(len(self.lrs)), self.lrs)

    def plot_mom_sched(self):
        plt.ylabel("Learning Rate")
        plt.xlabel("Iter")
        plt.plot(range(len(self.moms)), self.moms)









lrf = LRFinder(1e-5, 10, train_dataloader)
model = create_model(37)
model.to(device)
adam = optim.AdamW(model.parameters())
loss_func = nn.CrossEntropyLoss()  # loss function
print(lrf)
print(lrf.__dict__.keys())
object_methods = [method_name for method_name in dir(optim.Adam)
              if callable(getattr(optim.Adam, method_name))]
print(object_methods)


dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
resnet=create_model(37)
resnet.to("cuda")
loss_func = nn.CrossEntropyLoss()
adam=optim.AdamW(resnet.parameters(), lr=1e-3)
epochs=4
total = 0
correct = 0
train_loss = 0
test_loss = 0
best_acc = 0
trn_losses = []
trn_accs = []
val_losses = []
val_accs = []
preds=[]
n = np.ceil(len(train_dataloader.dataset)/bs)
n



ocp= OneCyclePolicy(epochs, 92, 2e-3)
ocp.fit(epochs, resnet, dataloaders_dict, loss_func, adam, True)









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

