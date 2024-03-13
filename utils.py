import os
from pathlib import Path
import time

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tensorly as tl
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import v2

def tic():
    return time.time()
def toc(tstart, name="Operation", printtime=True):
    dt = time.time() - tstart
    if printtime:
        print("%s took: %s sec.\n" % (name, dt))
    return dt

T_randrot = v2.Compose([
    v2.PILToTensor(),
    v2.RandomRotation(45),
    v2.ToPILImage()
])

T_tensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((28, 28))
])

T_randrottensor = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((28, 28)),
    v2.RandomRotation(45),
])

allclasses = []
for i in range(ord('0'), ord('9') + 1):
    allclasses.append(chr(i))
for i in range(ord('a'), ord('z') + 1):
    allclasses.append(chr(i))
class_to_idx_map = {chrclass: idx for idx, chrclass in enumerate(allclasses)}

def emnistletter_idx_to_class(idx):
    """
    emnist letters go from 1-26
    """
    return chr(idx - 1 + ord('a'))

def emnistdigit_idx_to_class(idx):
    """
    emnist digits go from 0-9
    """
    return chr(idx + ord('0'))

def emnistletter_idx_to_allidx(idx):
    """
    Converts emnist letter classes to our aggregate digit+letter classes.

    Digit letter classes are already 0-9, and at the our aggregate list has the same.
    """
    return idx + 9

def class_to_idx(chrcls):
    chrcls = str(chrcls)
    return class_to_idx_map[chrcls]

def idx_to_class(idx):
    return allclasses[idx]

class CharDataset(Dataset):
    """
    General dataset for single character data loading.

    Assumes that in the directory given, the subdirectories are the classes.
    These directory names should be the single character class.
    """
    def __init__(self, img_dir, transform=None, label_transform=None):
        self.img_dir = Path(img_dir)
        if not self.img_dir.exists():
            raise FileNotFoundError(f"{img_dir} does not exist")
        alldirs = [p for p in self.img_dir.glob("*") if p.is_dir()]
        self.imgpaths = []
        self.imglabels = []
        for d in alldirs:
            chclass = d.stem.lower()
            imgpaths_dirty = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            imgpaths = []
            for i in range(len(imgpaths_dirty)):
                # cv2 imread is ~3x faster than Image.open
                cv2opened = cv2.imread(str(imgpaths_dirty[i]))
                if cv2opened is None:
                    print(f"Image {imgpaths_dirty[i]} is not a valid image, skipping")
                    continue
                imgpaths.append(imgpaths_dirty[i])
            self.imgpaths.extend(imgpaths)
            self.imglabels.extend([chclass] * len(imgpaths))
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.imglabels)

    def __getitem__(self, idx):
        imgpath = self.imgpaths[idx]
        img = Image.open(imgpath).convert("L")
        label = self.imglabels[idx]
        if self.transform:
            img = self.transform(img)
        if self.label_transform:
            label = self.label_transform(label)
        return img, label
    
def ds_to_vectors(ds, size=28):
    imgs = np.array([np.array(x[0].resize((28, 28))).reshape(-1) for x in ds])
    labs = np.array([x[1] for x in ds])
    return imgs, labs

class GenericCharCNN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.encoder(x)
        labels_hat = torch.argmax(out, dim=1)
        loss = nn.functional.cross_entropy(out, y)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.encoder(x)
        labels_hat = torch.argmax(out, dim=1)
        loss = nn.functional.cross_entropy(out, y)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


class GenericCharCNN_nopool(GenericCharCNN):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
