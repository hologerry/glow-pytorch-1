import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .base import make_dataset_base_image_attr


class ExploDataset(Dataset):
    def __init__(self, root_dir, image_size=64, phase='train'):
        super().__init__()

        assert phase in ['train', 'test'], NotImplementedError

        assert os.path.isdir(root_dir), f"explo dataset {root_dir} does not exist"
        self.root_dir = root_dir

        self.base_dir = os.path.join(self.root_dir, 'base')
        assert os.path.exists(self.base_dir), f"explo dataset {self.base_dir} does not exist"

        self.datas = make_dataset_base_image_attr(self.base_dir, self.root_dir, "attributes.txt")
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
        data = dataset[index]

        base = self.transform(Image.open(data['base']).convert('RGB'))
        image = self.transform(Image.open(data['image']).convert('RGB'))
        font = torch.LongTensor(data['font'])
        char = torch.LongTensor(data['char'])
        attr = torch.FloatTensor(data['attr'])

        return {
            'base': base,
            'image': image,
            'font': font,
            'char': char,
            'attr': attr,
        }

    def __len__(self):
        return self.num_images
