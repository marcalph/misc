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
from albumentations import (CLAHE, Blur, Compose, Flip, GaussNoise,
                            GridDistortion, HorizontalFlip, HueSaturationValue,
                            IAAAdditiveGaussianNoise, IAAEmboss,
                            IAAPerspective, IAAPiecewiseAffine, IAASharpen,
                            MedianBlur, MotionBlur, OneOf, OpticalDistortion,
                            RandomBrightnessContrast, RandomCrop,
                            RandomRotate90, Resize, Rotate, ShiftScaleRotate,
                            Transpose)
from albumentations.pytorch import ToTensor
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

# todo use seaborn graphs

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

    def search(self, model, optimizer, loss_func):
        running_loss = 0.
        beta = 0.                                                   # smoothing param
        global device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def update_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def accuracy(output, target, is_test=False):
    global total
    global correct
    global preds
    batch_size = output.shape[0]
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



class OneCyclePolicy():
    def __init__(self, cycle_len, step_per_epoch, max_lr, momentum_vals=(0.95, 0.85), up_phase=0.25, end_phase=0.1, lr_fold=10):
        self.step_per_epoch = step_per_epoch
        self.lr_fold = lr_fold
        self.max_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.up_phase = up_phase
        self.end_phase = end_phase
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
        annihilation_phase = int(self.end_phase*self.step_per_epoch*self.cycle_len)
        up_phase = int(up_prcnt*(1-self.end_phase)*self.step_per_epoch*self.cycle_len)
        down_phase = self.step_per_epoch*self.cycle_len-up_phase-annihilation_phase
        return cycle(np.hstack([np.linspace(self.max_lr/self.lr_fold, self.max_lr, up_phase),  # increase
                                np.linspace(self.max_lr, self.max_lr/self.lr_fold, down_phase),  # decrease
                                np.linspace(self.max_lr/self.lr_fold, self.max_lr/self.lr_fold/100, annihilation_phase)]))

    def mom_gen(self):
        down_prcnt = self.up_phase
        const_phase = int(self.end_phase*self.step_per_epoch*self.cycle_len)
        down_phase = int(down_prcnt*(1-self.end_phase)*self.step_per_epoch*self.cycle_len)
        up_phase = self.step_per_epoch*self.cycle_len-down_phase-const_phase
        return cycle(np.hstack([np.linspace(self.high_mom, self.low_mom, down_phase),  # decrease
                                np.linspace(self.low_mom, self.high_mom, up_phase),  # increase
                                np.repeat(self.high_mom, const_phase)]))

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
                update_lr(optimizer, lr)
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

    def validate(self, model, dataloader):
        model.eval()

        running_loss = 0.
        running_corrects = 0

        avg_beta = 0.98
        with torch.no_grad():
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
            self.validate(model, dataloaders["val"])
            print(f"{epoch+1:5}{self.trn_losses[-1]:15.5f}{self.val_losses[-1]:15.5f}{self.trn_accs[-1]:15.5f}{self.val_accs[-1]:15.5f}")

    def plot_lr_sched(self):
        plt.ylabel("Learning Rate")
        plt.xlabel("Iter")
        plt.plot(range(len(self.lrs)), self.lrs)

    def plot_mom_sched(self):
        plt.ylabel("Learning Rate")
        plt.xlabel("Iter")
        plt.plot(range(len(self.moms)), self.moms)



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




def create_model(num_classes):
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model, input_size





if __name__ == "__main__":
    path = pathlib.Path("data/oxford-iiit-pet")
    path_anno = path/'annotations'
    path_img = path/'images'

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

    train_ds = PetDataset(path_img, path_anno/"trainval.txt", train_transforms)
    train_dataloader = DataLoader(train_ds, batch_size=64,
                            shuffle=True, num_workers=4)

    val_ds = PetDataset(path_img, path_anno/"test.txt", train_transforms)
    val_dataloader = DataLoader(val_ds, batch_size=64,
                            shuffle=True, num_workers=4)

    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}


    lrf = LRFinder(1e-8, 10, train_dataloader)
    model, input_size = create_model(37)
    model.to("cuda")
    adam = optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()  # loss function

    # print(lrf)
    # print(lrf.__dict__.keys())
    # object_methods = [method_name for method_name in dir(optim.Adam)
    #             if callable(getattr(optim.Adam, method_name))]
    # print(object_methods)
    # lrf.search(model, adam, loss_func)
    # lrf.plot_search()
