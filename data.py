# -*- coding: utf-8 -*-
import os
import random
import numpy as np

import PIL
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import utils as utils
from pdb import set_trace
from fixmatch_data import TransformFixMatch
import seed
torch.manual_seed(seed.a)
torch.cuda.manual_seed(seed.a)
random.seed(seed.a)
np.random.seed(seed.a)

class_num = {"officehome": 65, "domainnet": 345, "cifar10": 10, "office31": 31, "domainnet_50": 50}

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # リサイズ
        return img

def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, domain, split, transforms, subset_idx = 0, with_strong=False):
        self.dataset = dataset
        self.domain = domain
        self.split = split
        self.subset_idx = subset_idx
        if self.dataset == "domainnet_50":
            self.txt_file = os.path.join('..', 'data', self.dataset, "train_val_test_split", f'{self.domain}_{self.subset_idx}_{self.split}.txt')
        else:
            self.txt_file = os.path.join('..', 'data', self.dataset, f'{self.domain}_{self.split}.txt')
        self.data, self.labels = None, None
        with open(self.txt_file, "r") as f:
            self.dataset_len = len(f.readlines())
        self.transforms = transforms
        self.with_strong = with_strong
        self.strong_tranforms = TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224)
        self.num_classes = class_num[dataset]

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.data is None:
            self.data, self.labels = make_dataset_fromlist(self.txt_file)
        image_path, label = self.data[idx], self.labels[idx]
        if self.dataset == "domainnet_50":
            datum = pil_loader(os.path.join('..', 'data', "domainnet", image_path))
        else:
            datum = pil_loader(os.path.join('..', 'data', self.dataset, image_path))
        if self.with_strong:
            return (self.transforms(datum), self.strong_tranforms(datum), int(label))
        return (self.transforms(datum), int(label))

    def __getitem_with_path__(self, idx):
        """
        画像パスも返す、テスト専用のgetitemメソッド
        """
        if self.data is None:
            self.data, self.labels = make_dataset_fromlist(self.txt_file)
        image_path, label = self.data[idx], self.labels[idx]
        datum = pil_loader(os.path.join('..', 'data', self.dataset, image_path))
        return self.transforms(datum), int(label), image_path

    def get_num_classes(self):
        return self.num_classes

# data.py または 適切なファイルに追加
class ImageDatasetWithPath(ImageDataset):
    def __getitem__(self, idx):
        if self.data is None:
            self.data, self.labels = make_dataset_fromlist(self.txt_file)
        image_path, label = self.data[idx], self.labels[idx]
        if self.dataset == "domainnet_50":
            datum = pil_loader(os.path.join('..', 'data', "domainnet", image_path))
        else:
            datum = pil_loader(os.path.join('..', 'data', self.dataset, image_path))
        if self.with_strong:
            return (self.transforms(datum), self.strong_tranforms(datum), int(label), image_path)
        return (self.transforms(datum), int(label), image_path)


def custom_collate_with_paths(batch):
    """
    カスタムコラテート関数：バッチ内の各サンプルを (data, target, path) としてバッチ化。
    """
    data = torch.stack([item[0] for item in batch], dim=0)
    target = torch.tensor([item[1] for item in batch])
    paths = [item[2] for item in batch]
    return data, target, paths

class ASDADataset:
    # Active Semi-supervised DA Dataset class
    def __init__(self, dataset, name, pair, subset_idx = 0, data_dir='data', valid_ratio=0.2, batch_size=128, augment=False):
        self.dataset = dataset
        self.name = name  # domain name
        self.pair = pair  # source/target
        self.subset_idx = subset_idx
        self.data_dir = data_dir
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.train_size = None
        self.train_dataset = None
        self.num_classes = class_num[dataset]

    def get_num_classes(self):
        return self.num_classes

    def get_dsets(self, normalize=True, apply_transforms=False):
        if self.dataset in ["domainnet", "office31", "officehome", "domainnet_50"]:
            assert self.name in ["real", "quickdraw", "sketch", "infograph", "clipart", "painting", "amazon", "webcam", "dslr", "Art", "Clipart", "Product", "RealWorld"]
            normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
                if normalize else transforms.Normalize([0, 0, 0], [1, 1, 1])
            if apply_transforms:
                data_transforms = {
                    'train': transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize_transform
                    ]),
                }
            else:
                data_transforms = {
                    'train': transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        normalize_transform
                    ]),
                }
            data_transforms['test'] = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize_transform
            ])
            train_dataset = ImageDataset(self.dataset, self.name, 'train', data_transforms['train'], subset_idx = self.subset_idx)
            val_dataset = ImageDataset(self.dataset, self.name, 'val', data_transforms['test'], subset_idx = self.subset_idx) if self.pair == "source" else None
            test_dataset = ImageDataset(self.dataset, self.name, 'test', data_transforms['test'], subset_idx = self.subset_idx)
            self.num_classes = train_dataset.get_num_classes()
        else:
            raise NotImplementedError

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        return train_dataset, val_dataset, test_dataset

    def get_dsets_with_paths(self, normalize=True, apply_transforms=False):
        # 共通の正規化変換
        normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalize else transforms.Normalize([0, 0, 0], [1, 1, 1])

        if apply_transforms:
            print("applied transform to",self.name,self.pair)
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_transform
                ]),
            }
        else:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    normalize_transform
                ]),
            }
        data_transforms['test'] = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize_transform
        ])

        # ImageDatasetWithPathを使用してデータセットを作成
        train_dataset = ImageDatasetWithPath(self.dataset, self.name, 'train', data_transforms['train'], subset_idx=self.subset_idx)
        val_dataset = ImageDatasetWithPath(self.dataset, self.name, 'val', data_transforms['test'], subset_idx=self.subset_idx) if self.pair == "source" else None
        test_dataset = ImageDatasetWithPath(self.dataset, self.name, 'test', data_transforms['test'], subset_idx=self.subset_idx)
        
        self.num_classes = train_dataset.get_num_classes()

        return train_dataset, val_dataset, test_dataset

    def get_loaders(self, num_workers=4, normalize=True):
        if not self.train_dataset:
            self.get_dsets(normalize=normalize)

        num_train = len(self.train_dataset)
        self.train_size = num_train

        if self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart", "amazon", "webcam", "dslr", "Art", "Clipart", "Product", "RealWorld"]:
            train_idx = np.arange(len(self.train_dataset))
            train_sampler = SubsetRandomSampler(train_idx)
        else:
            raise NotImplementedError

        train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler,
                                                   batch_size=self.batch_size, num_workers=num_workers)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size) if self.val_dataset is not None else None
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

        # train_idx = np.arange(10)  # train_datasetのサンプル数が10の場合 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        return train_loader, val_loader, test_loader, train_idx

    def get_non_shuffling_loaders_with_paths(self, num_workers=4, normalize=True):
        if not self.train_dataset:
            self.get_dsets(normalize=normalize, apply_transforms=False)

        # ImageDatasetWithPathのインスタンスを作成
        train_dataset_with_path = ImageDatasetWithPath(
            self.dataset,
            self.name,
            'train',
            self.train_dataset.transforms,
            subset_idx=self.subset_idx,
            with_strong=self.train_dataset.with_strong
        )

        test_dataset_with_path = ImageDatasetWithPath(
            self.dataset,
            self.name,
            'test',
            self.test_dataset.transforms,
            subset_idx=self.subset_idx
        )

        val_dataset_with_path = None
        if self.val_dataset is not None:
            val_dataset_with_path = ImageDatasetWithPath(
                self.dataset,
                self.name,
                'val',
                self.val_dataset.transforms,
                subset_idx=self.subset_idx
            )

        # DataLoaderの作成（シャッフルなし）
        train_loader = torch.utils.data.DataLoader(
            train_dataset_with_path,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=custom_collate_with_paths
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset_with_path,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=custom_collate_with_paths
        )

        val_loader = None
        if val_dataset_with_path is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset_with_path,
                batch_size=self.batch_size,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=custom_collate_with_paths
            )

        return train_loader, val_loader, test_loader

    
    def get_non_shuffling_train_loader_with_paths(self, num_workers=4, normalize=True):
        if not self.train_dataset:
            self.get_dsets(normalize=normalize)
        
        # 新しいデータセットインスタンスを作成
        train_dataset_with_path = ImageDatasetWithPath(
            self.dataset,
            self.name,
            'train',
            self.train_dataset.transforms,  # 同じトランスフォームを使用
            subset_idx=self.subset_idx,
            with_strong=self.train_dataset.with_strong
        )
        
        num_train = len(train_dataset_with_path)
        self.train_size = num_train

        if self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart", "amazon", "webcam", "dslr", "Art", "Clipart", "Product", "RealWorld"]:
            train_idx = np.arange(len(train_dataset_with_path))
            # シャッフルしない DataLoader を作成
            train_loader = torch.utils.data.DataLoader(
                train_dataset_with_path,
                batch_size=self.batch_size,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=custom_collate_with_paths  # カスタムコラテート関数を指定
            )
        else:
            raise NotImplementedError

        return train_loader, train_idx


if __name__ == "__main__":
    pass