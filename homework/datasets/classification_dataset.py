import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import torch.nn as nn

LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


class SuperTuxDataset(Dataset):
    """
    SuperTux dataset for classification
    """

    def __init__(
        self,
        dataset_path: str,
        transform_pipeline: str = "default",
    ):
        self.transform = self.get_transform(transform_pipeline)
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    img_path = Path(dataset_path, fname)
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((img_path, label_id))

    def get_transform(self, transform_pipeline: str = "default"):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.ToTensor()
        elif transform_pipeline == "aug":
            # TODO: construct your custom augmentation
            # Custom augmentation pipeline
            xform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
                transforms.RandomRotation(degrees=15),   # Random rotation within +/- 15 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2788, 0.2657, 0.2629], std=[0.2064, 0.1944, 0.2252]),
            ])


        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Pairs of images and labels (int) for classification
        """
        img_path, label_id = self.data[idx]
        img = Image.open(img_path)
        data = (self.transform(img), label_id)

        return data


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 4,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    """
    Constructs the dataset/dataloader.
    The specified transform_pipeline must be implemented in the SuperTuxDataset class.

    Args:
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        DataLoader or Dataset
    """
    dataset = SuperTuxDataset(dataset_path, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        return nn.CrossEntropyLoss()(logits, target)



def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
        """
        Arguments:
            outputs: torch.Tensor, shape (b, num_classes) either logits or probabilities
            labels: torch.Tensor, shape (b,) with the ground truth class labels

        Returns:
            a single torch.Tensor scalar
        """
        outputs_idx = outputs.max(1)[1].type_as(labels)

        return (outputs_idx == labels).float().mean()
