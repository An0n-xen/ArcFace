import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    def __init__(self, lfw_data, transform=None):
        self.images = lfw_data.images
        self.targets = lfw_data.target
        self.target_names = lfw_data.target_names
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = torch.tensor(self.targets[idx]).long()

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        if self.transform:
            image = self.transform(image)

        return image, target
