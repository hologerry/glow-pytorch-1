import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

ATTR_ANNO = "attributes.txt"


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_images_and_annotation(root_dir):
    images = {}
    attr_file = None
    assert os.path.isdir(root_dir), f"{root_dir} does not exist"
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images[os.path.splitext(fname)[0]] = path
            elif fname.lowe() == ATTR_ANNO:
                attr_file = os.path.join(root, fname)

    assert attr_file is not None, f"Failed to find {ATTR_ANNO}"

    # parse all image
    print("Parsing all images and their attributes...")
    datas = []
    with open(attr_file, "r") as f:
        attr_names = []
        lines = f.readlines()
        attr_names = lines[0].split(" ")
        for idx, line in enumerate(lines[1:]):
            line = line.strip()
            line = line.split("  ")
            fname = os.path.splitext(line[0])[0]
            font = idx // 52
            char = int(fname.split('_')[1].split('.')[0])
            attr_vals = [(float(v)/100.0) for v in line[1:]]
            assert len(attr_vals) == len(attr_names), f"{fname} has only {len(attr_vals)} attributes"
            datas.append({
                "image": images[fname],
                "font": font,
                "char": char,
                "attr": attr_vals
            })
    print(f"Found {len(datas)} images with font, char and attributes label.")


class ExploDataset(Dataset):
    def __init__(self, root_dir, image_size=64, phase='train'):
        super().__init__()

        assert phase in ['train', 'test'], NotImplementedError
        self.phase = phase

        assert os.path.isdir(root_dir), f"{root_dir} does not exist"
        self.root_dir = root_dir

        self.datas = find_images_and_annotation(self.root_dir)
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
        data_B = dataset[random.randint(0, self.num_images)]

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
