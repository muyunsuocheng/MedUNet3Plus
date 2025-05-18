# data/dataloader.py
import os
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib

class MedicalDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.samples = self._load_samples()

    def _load_samples(self):
        """加载划分好的数据列表"""
        split_file = os.path.join(self.data_dir, f"{self.split}_list.txt")
        with open(split_file, "r") as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name = self.samples[idx]
        image_path = os.path.join(self.data_dir, "images", file_name)
        label_path = os.path.join(self.data_dir, "labels", file_name)

        # 加载NIfTI文件
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # 转换为Tensor并添加通道维度
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return {"image": image, "label": label}

def get_loaders(data_dir, batch_size=4):
    """获取数据加载器"""
    train_dataset = MedicalDataset(data_dir, split="train")
    val_dataset = MedicalDataset(data_dir, split="val")
    test_dataset = MedicalDataset(data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    return train_loader, val_loader, test_loader