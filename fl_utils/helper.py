import sys
sys.path.append("../")
import os

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image

from collections import defaultdict
import random
import numpy as np
from models.resnet import ResNet18

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {
        }
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {
            i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {
            classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {
        }
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {
            classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {
            i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

class Helper:
    def __init__(self, config):
        self.config = config
        
        if self.config.dataset in ['cifar10','gtsrb']:
            self.config.data_folder = './datasets'
        elif self.config.dataset == 'TinyImagenet':
            self.config.data_folder = './datasets/tiny-imagenet-200'
        self.local_model = None
        self.global_model = None
        self.client_models = []
        self.setup_all()

    def setup_all(self):
        self.load_data()
        self.load_model()
        self.config_adversaries()

    def load_model(self):
        self.local_model = ResNet18(num_classes = self.num_classes)
        self.local_model.cuda()
        self.global_model = ResNet18(num_classes = self.num_classes)
        self.global_model.cuda()
        for i in range(self.config.num_total_participants):
            t_model = ResNet18(num_classes = self.num_classes)
            t_model.cuda()
            self.client_models.append(t_model)
        
    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers=self.config.num_worker)
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)

        return test_loader

    def load_data(self):
        if self.config.dataset == 'cifar10':
            self.num_classes = 10
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.train_dataset = datasets.CIFAR10(
                self.config.data_folder, train=True, 
                download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(
                self.config.data_folder, train=False, transform=transform_test)
        elif self.config.dataset == 'TinyImagenet':
            self.num_classes = 200
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.train_dataset = TinyImageNet(
                self.config.data_folder, train=True, 
                transform=transform_train)
            self.test_dataset = TinyImageNet(
                self.config.data_folder, train=False, transform=transform_test)
        elif self.config.dataset == 'gtsrb':
            self.num_classes = 43
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
            ])
            self.train_dataset = datasets.GTSRB(
                self.config.data_folder, split='train', 
                download=True, transform=transform_train)
            self.test_dataset = datasets.GTSRB(
                self.config.data_folder, download = True, split='test', transform=transform_test)
        indices_per_participant = self.sample_dirichlet_train_data(
            self.config.num_total_participants,
            alpha=self.config.dirichlet_alpha)
        
        train_loaders = [self.get_train(indices) 
            for pos, indices in indices_per_participant.items()]

        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_worker)
    
    def config_adversaries(self):
        if self.config.is_poison:
            self.adversary_list = list(range(self.config.num_adversaries))
        else:
            self.adversary_list = list()