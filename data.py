from typing import Tuple
from torchvision import datasets, transforms
from partition import dirichlet_partition, split_client_train_val, compute_kl_heterogeneity_scores, normalize_scores


def load_dataset(dataset: str, data_dir: str):
    dataset = dataset.lower()
    if dataset == "cifar10":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
        labels = train_set.targets
        return train_set, test_set, labels

    if dataset == "fashionmnist":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_tf)
        test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_tf)
        labels = train_set.targets.numpy().tolist()
        return train_set, test_set, labels

    raise ValueError(f"Unsupported dataset: {dataset}")


def prepare_federated_data(config):
    train_set, test_set, labels = load_dataset(config.dataset, config.data_dir)
    client_indices, class_counts = dirichlet_partition(
        labels=labels,
        num_clients=config.num_clients,
        alpha=config.dirichlet_alpha,
        num_classes=config.num_classes,
        min_size=config.min_client_size,
        seed=config.seed,
    )
    train_indices, val_indices = split_client_train_val(
        client_indices, val_fraction=config.val_fraction, seed=config.seed + 7
    )
    hetero_scores = normalize_scores(compute_kl_heterogeneity_scores(class_counts))
    return train_set, test_set, train_indices, val_indices, class_counts, hetero_scores
