import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .base import find_images_and_annotation


class ExploABDataset(Dataset):
    def __init__(self, root_dir, image_size=64, phase='train'):
        super().__init__()

        assert phase in ['train', 'test'], NotImplementedError
        self.phase = phase

        assert os.path.isdir(root_dir), f"{root_dir} does not exist"
        self.root_dir = root_dir

        self.datas = find_images_and_annotation(self.root_dir, "attributes.txt")
        self.dataset_size = len(self.datas)
        self.train_size = 52 * 120  # train_char * train_font
        self.train_set = self.datas[:self.train_size]
        self.test_set = self.datas[self.train_set:]

        self.num_images = len(self.train_set) if self.phase == 'train' else len(self.test_set)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

    def __getitem__(self, index):
        dataset = self.train_set if self.phase == 'train' else self.test_set
        data_A = dataset[index]
        data_B = dataset[(index + 52 * random.randint(0, 148)) % self.num_images]

        image_A = self.transform(Image.open(data_A['image']).convert('RGB'))
        font_A = torch.LongTensor(data_A['font'])
        char_A = torch.LongTensor(data_A['char'])
        attr_A = torch.FloatTensor(data_A['attr'])

        image_B = self.transform(Image.open(data_B['image']).convert('RGB'))
        font_B = torch.LongTensor(data_B['font'])
        char_B = torch.LongTensor(data_B['char'])
        attr_B = torch.FloatTensor(data_B['attr'])

        return {
            'img_A': image_A,
            'font_A': font_A,
            'char_A': char_A,
            'attr_A': attr_A,
            'img_B': image_B,
            'font_B': font_B,
            'char_B': char_B,
            'attr_B': attr_B,
        }

    def __len__(self):
        return self.num_images
