import os
from pathlib import Path

import lightning as L
import torch
from torchvision.datasets import EMNIST
from torch.utils.data import random_split

import utils

torch.set_float32_matmul_precision("high")

# this dataset is already random rotated
handiso_ds_train = utils.CharDataset("data/handwritten-isolated-english/train", transform=utils.T_tensor, label_transform=utils.class_to_idx)
handiso_ds_test = utils.CharDataset("data/handwritten-isolated-english/test", transform=utils.T_tensor, label_transform=utils.class_to_idx)

notmnist_ds = utils.CharDataset("data/notMNIST_small", transform=utils.T_randrottensor, label_transform=utils.class_to_idx)
notmnist_ds_train, notmnist_ds_test = random_split(notmnist_ds, [0.8, 0.2])

stdocr_ds_train_orig = (
    utils.CharDataset("data/standard_ocr_dataset/data/training_data", transform=utils.T_randrottensor, label_transform=utils.class_to_idx) +
    utils.CharDataset("data/standard_ocr_dataset/data2/training_data", transform=utils.T_randrottensor, label_transform=utils.class_to_idx)
)
stdocr_ds_test = (
    utils.CharDataset("data/standard_ocr_dataset/data/testing_data", transform=utils.T_randrottensor, label_transform=utils.class_to_idx) +
    utils.CharDataset("data/standard_ocr_dataset/data2/testing_data", transform=utils.T_randrottensor, label_transform=utils.class_to_idx)
)

mnist_ds_train_letters = EMNIST(Path(os.getcwd(), "data"), "letters", download=True, train=True, transform=utils.T_randrottensor, target_transform=utils.emnistletter_idx_to_allidx)
mnist_ds_train_digits = EMNIST(Path(os.getcwd(), "data"), "digits", download=True, train=True, transform=utils.T_randrottensor)
mnist_ds_train_orig = mnist_ds_train_letters + mnist_ds_train_digits
mnist_ds_test_letters = EMNIST(Path(os.getcwd(), "data"), "letters", download=True, train=False, transform=utils.T_randrottensor, target_transform=utils.emnistletter_idx_to_allidx)
mnist_ds_test_digits = EMNIST(Path(os.getcwd(), "data"), "digits", download=True, train=False, transform=utils.T_randrottensor)
mnist_ds_test_orig = mnist_ds_test_letters + mnist_ds_test_digits

stdocr_ds_train, _ = random_split(stdocr_ds_train_orig, [0.5, 0.5])
mnist_ds_train, _ = random_split(mnist_ds_train_orig, [0.05, 0.95])
mnist_ds_test, _ = random_split(mnist_ds_test_orig, [0.05, 0.95])

consolidated_ds_train = handiso_ds_train + notmnist_ds_train + stdocr_ds_train + mnist_ds_train
consolidated_ds_test = handiso_ds_test + notmnist_ds_test + stdocr_ds_test + mnist_ds_test
trainloader = torch.utils.data.DataLoader(consolidated_ds_train, shuffle=True, num_workers=31, batch_size=32)
testloader = torch.utils.data.DataLoader(consolidated_ds_test, shuffle=True, num_workers=31, batch_size=32)


# genCNN = utils.GenericCharCNN()
genCNN = utils.GenericCharCNN_nopool()
trainer = L.Trainer(max_epochs=20)
trainer.fit(model=genCNN, train_dataloaders=trainloader)

trainer.test(genCNN, dataloaders=testloader)
