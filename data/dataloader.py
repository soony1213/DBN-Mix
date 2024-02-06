import re
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data.autoaugment import autoSVHNPolicy, autoCIFARPolicy, autoImageNetPolicy
from data.randaugment import randSVHNPolicy, randCIFARPolicy, randImageNetPolicy
from data.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import pdb

dataset_statistics = {
    'cifar10': {
        'mean' : (0.4914, 0.4822, 0.4465),
        'std' : (0.2471, 0.2435, 0.2616)
    },
    'cifar100' : {
        'mean' :  (0.5071, 0.4867, 0.4408),
        'std' : (0.2675, 0.2565, 0.2761)
    },
    'imagenetLT' : {
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225)
    },
    'iNaturalist18' : {
        'mean' : (0.466, 0.471, 0.380),
        'std' : (0.195, 0.194, 0.192)
    },
    'placesLT' : {
        'mean' : (0.485, 0.456, 0.406),
        'std' : (0.229, 0.224, 0.225)
    }
}
num_classes_list = {
    'cifar10' : 10,
    'cifar100' : 100,
    'placesLT' : 365,
    'imagenetLT' : 1000,
    'iNaturalist18' : 8142
}

def get_num_classes(dataset):
    return num_classes_list[dataset]

def load_data(data_root, dataset, batch_size, test_open=False, num_workers=4, ngpus_per_node=None, args=None):
    dataset_mean, dataset_std = dataset_statistics[dataset]['mean'], dataset_statistics[dataset]['std']
    data_root = data_root + '/' + dataset
    if dataset in ['cifar10', 'cifar100']:
        train_transform_s = get_cifar_transform('strong_train', args.augmentation, dataset_mean, dataset_std)
        train_transform_w = get_cifar_transform('weak_train', args.augmentation, dataset_mean, dataset_std)
        val_transform = get_cifar_transform('test', args.augmentation, dataset_mean, dataset_std)
        test_transform = get_cifar_transform('test', args.augmentation, dataset_mean, dataset_std)
    else:
        raise ValueError
    num_classes = num_classes_list[dataset]

    if dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root=data_root, args=args, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                         rand_number=args.seed, train=True, batch_size=batch_size,
                                         download=True, transform=train_transform_w, strong_transform=train_transform_s)
        val_dataset = datasets.CIFAR10(root=data_root, train=False, download=True,
                                       transform=val_transform)
        inference_dataset = None
        test_dataset = None
    elif dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root=data_root, args=args, imb_type=args.imb_type, imb_factor=args.imb_factor,
                                          rand_number=args.seed, train=True, batch_size=batch_size,
                                          download=True, transform=train_transform_w, strong_transform=train_transform_s)
        val_dataset = datasets.CIFAR100(root=data_root, train=False, download=True,
                                        transform=val_transform)
        inference_dataset = None
        test_dataset = None
    else:
        import warnings
        warnings.warn('Dataset is not listed')
        return

    if ngpus_per_node > 1:
        drop_last = True
    else:
        drop_last = False

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    
    if inference_dataset:
        inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=100, shuffle=False, 
        num_workers=num_workers, pin_memory=True, drop_last=False)
    
    if dataset in ['cifar10', 'cifar100']:
        if inference_dataset:
            return train_loader, val_loader, inference_loader
        else:
            return train_loader, val_loader
    else:
        if inference_dataset:
            return train_loader, val_loader, test_loader, inference_loader
        else:
            return train_loader, val_loader, test_loader

class noPolicy(object):
    def __call__(self, img):
        return img

def get_augment(dataset, aug_method):
    if dataset == 'svhn':
        if aug_method == 'autoaugment':
            return autoSVHNPolicy
        elif aug_method == 'randaugment':
            return randSVHNPolicy
        elif aug_method == 'default':
            return noPolicy
        else:
            raise ValueError
    if dataset == 'cifar':
        if aug_method == 'autoaugment':
            return autoCIFARPolicy
        elif aug_method == 'randaugment':
            return randCIFARPolicy
        elif aug_method == 'default':
            return noPolicy
        else:
            raise ValueError
    if dataset in ['imagenetLT', 'iNaturalist18', 'placesLT']:
        if aug_method == 'augtoaugment':
            return autoImageNetPolicy
        elif aug_method == 'randaugment':
            return randImageNetPolicy
        elif aug_method == 'default':
            return noPolicy
        else:
            raise ValueError

def get_cifar_transform(split, aug_method, dataset_mean, dataset_std):
    data_transforms = {
        'strong_train' : transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            get_augment('cifar', aug_method)(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ]),
        'weak_train' : transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
        ]),
        'val' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std),
        ])
    }
    return data_transforms[split]

def get_imagenet_transforms(split, aug_method, dataset_mean, dataset_std):
    data_transforms = {
        'strong_train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            get_augment('imagenetLT', aug_method)(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]) if aug_method else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'weak_train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'val' :  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])
    }
    return data_transforms[split]

def get_places_transforms(split, aug_method, dataset_mean, dataset_std):
    data_transforms = {
        'strong_train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            get_augment('placesLT', aug_method)(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]) if aug_method else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'weak_train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'val' :  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])
    }
    return data_transforms[split]

def get_iNaturalist_transforms(split, aug_method, dataset_mean, dataset_std):
    data_transforms = {
        'strong_train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            get_augment('iNaturalist18', aug_method)(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]) if aug_method else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'weak_train' : transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'val' :  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ]),
        'test' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])
    }
    return data_transforms[split]