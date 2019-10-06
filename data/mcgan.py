import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .base import make_dataset


class MCGANDataset(Dataset):
    def __init__(self, root_dir, image_size=64, phase='train'):
        super().__init__()

        self.phase = phase
        assert phase in ['train', 'val', 'test'], NotImplementedError

        self.root_dir = root_dir
        assert os.path.isdir(root_dir), f"mcgan dataset {root_dir} does not exist"

        self.phase_dir = os.path.join(self.root_dir, phase)
        self.base_dir = os.path.join(self.root_dir, 'base')
        assert os.path.exists(self.phase_dir), f"mcgan dataset {self.phase_dir} does not exist"
        assert os.path.exists(self.base_dir), f"mcgan dataset {self.base_dir} does not exist"

        self.images = sorted(make_dataset(self.phase_dir))
        self.num_images = len(self.images)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ])

    def __getitem__(self, index):
        image_path = self.images[index]
        char = image_path[-5]
        base_image_path = os.path.join(self.base_dir, char+'.png')
        image = self.transform(Image.open(image_path).convert('RGB'))
        base_image = self.transform(Image.open(base_image_path).convert('RGB'))

        return {
            'img': image,
            'base': base_image
        }

    def __len__(self):
        return self.num_images
