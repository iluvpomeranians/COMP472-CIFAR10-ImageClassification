import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_cifar10(root="./data/raw"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),                    # Resize to 224x224 for models like ResNet
        transforms.ToTensor(),                            # This converts NumPy array to PyTorch Tensor [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means and stds
                             std=[0.229, 0.224, 0.225])
    ])
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    # keep first 500 train + 100 test per class
    def subset_by_class(dataset, n_per_class):
        targets = torch.tensor(dataset.targets)                                 # targets is a list of labels
        indices = []
        for cls in range(10):                                                   #cls means class index (e.g. 0 = airplane, 1 = automobile, etc.)
            cls_idx = (targets == cls).nonzero(as_tuple=True)[0][:n_per_class]  # So tagets is a 1-D tensor and we try to match the label cls by returning
            indices.extend(cls_idx.tolist())                                    # a boolean tensor like '[False, False, True, False, ...]' --> extract a tensor of indices:
        subset = torch.utils.data.Subset(dataset, indices)                      # tensor([2, 0, 7, ...])
        return subset

    return subset_by_class(train_set, 500), subset_by_class(test_set, 100)


def load_cifar10_vgg(batch_size=128):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root="./data/raw", train=True, download=True, transform=transform)
    test_set  = datasets.CIFAR10(root="./data/raw", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
