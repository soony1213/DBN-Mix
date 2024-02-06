import math
import random
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.utils.extmath import softmax

import pdb

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    subset_thr = [500]

    def __init__(self, root, args, imb_type='exp', imb_factor=0.01, rand_number=0, train=True, batch_size=128,
                 transform=None, strong_transform=None, target_transform=None, download=False, subset_thr=[100, 20]):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.train = train
        self.gen_imbalanced_data(img_num_list)
        self.args = args
        self.dual_sample = self.args.dual_sample
        self.strong_transform = strong_transform
        self.subset_thr = subset_thr
        self.batch_size = batch_size
        if self.dual_sample:
            self.class_weight, self.sum_weight, self.soft_class_weight, self.soft_sum_weight, self.num_list = self.get_weight(self.get_annotations(), self.cls_num, self.args.weight_gamma)
            self.class_dict = self._get_class_dict()
            self.class_weight_bal = [sum(self.class_weight)/self.cls_num for _ in range(self.cls_num)]
            self.current_epoch = 0
            self.trigger = 0
            self.iter_per_epoch = math.ceil(len(self)/self.batch_size)
            self.instance_weights = np.ones(len(self.data))

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sample_class_index_by_weight(self, use_soft_weight):
        class_weight = self.class_weight
        sum_weight = self.sum_weight
        rand_number, now_sum = random.random() * sum_weight, 0
        for i in range(self.cls_num):
            now_sum += class_weight[i]
            if rand_number <= now_sum:
                return i

    def sample_class_index_by_weight_forward(self, use_soft_weight=None):
        class_weight = self.class_weight[::-1]
        sum_weight = self.sum_weight
        cls_num = self.cls_num
        rand_number, now_sum = random.random() * sum_weight, 0
        for i in range(cls_num):
            now_sum += class_weight[i]
            if rand_number <= now_sum:
                return i

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
        # return self.num_list

    def get_weight(self, annotations, num_classes, weight_gamma):
        num_list = [0] * num_classes
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
        max_num = max(num_list)
        if self.args.weight_type == 'freq':
            class_weight = [max_num / i for i in num_list]
        elif self.args.weight_type == 'cb':
                if self.args.cb_beta == 0:
                    cb_beta = (sum(self.num_list) - 1) / sum(self.num_list)
                else:
                    cb_beta = self.args.cb_beta
                class_weight = [(1-cb_beta)/(1-cb_beta**number) for number in num_list]
        sum_weight = sum(class_weight)
        soft_class_weight = [val ** (1 / weight_gamma) for val in class_weight]
        soft_sum_weight = sum(soft_class_weight)
        return class_weight, sum_weight, soft_class_weight, soft_sum_weight, num_list

    def update_weights(self, update_val, index, update_ema=None):
        if update_ema:
            self.instance_weights[index] = update_ema * self.instance_weights[index] + (1-update_ema)*update_val.cpu().numpy()
        else:
            self.instance_weights[index] = update_val.cpu().numpy()

    def generate_weight(self, index, args, sample_class=None):
        weight = self.instance_weights[index]
        return weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index from sampler
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        ############## first sampler ##############
        # self.tic_toc()
        if self.args.sampler_type == 'default':
            if self.args.use_soft_weight:
                sample_class_a = self.sample_class_index_by_weight_forward(use_soft_weight=self.args.use_soft_weight)
                sample_indexes = self.class_dict[sample_class_a]
                sample_index_a = np.random.choice(sample_indexes)
            else:
                sample_index_a = index
                sample_class_a = self.targets[sample_index_a]
        elif self.args.sampler_type in ['balance', 'reverse']:
            if self.args.sampler_type  == 'balance':
                sample_class_a = random.randint(0, self.cls_num - 1)
            elif self.args.sampler_type  == 'reverse':
                sample_class_a = self.sample_class_index_by_weight(use_soft_weight=self.args.use_soft_weight)
            sample_indexes = self.class_dict[sample_class_a]
            sample_index_a = np.random.choice(sample_indexes)
        else:
            assert -1

        img, target = self.data[sample_index_a], self.targets[sample_index_a]
        img = Image.fromarray(img)

        ############## dual sampler ##############
        if self.args.dual_sample:
            assert self.args.dual_sampler_type in ['default', 'balance', 'reverse', 'uniform']
            if self.args.dual_sampler_type == 'default':
                if self.args.use_dual_soft_weight:
                    sample_class_b = self.sample_class_index_by_weight_forward(use_soft_weight=self.args.use_dual_soft_weight)
                    sample_indexes = self.class_dict[sample_class_b]
                    weight = self.generate_weight(index=sample_indexes, args=self.args,
                                                  sample_class=[sample_class_a, sample_class_b])
                    sample_indexes, weight = self.instance_subset(sample_indexes, weight, self.args.ins_ratio)
                    sample_index_b = np.random.choice(sample_indexes, p=np.exp(weight) / sum(np.exp(weight)))
                else:
                    sample_index_b = index
            elif self.args.dual_sampler_type in ['balance', 'reverse']:
                if self.args.dual_sampler_type == 'balance':
                    sample_class_b = random.randint(0, self.cls_num - 1)
                elif self.args.dual_sampler_type == 'reverse':
                    sample_class_b = self.sample_class_index_by_weight(use_soft_weight=self.args.use_dual_soft_weight)
                sample_indexes = self.class_dict[sample_class_b]
                weight = self.generate_weight(index=sample_indexes, args=self.args,
                                              sample_class=[sample_class_a, sample_class_b])
                sample_index_b = np.random.choice(sample_indexes, p=np.exp(weight) / sum(np.exp(weight)))
            elif self.args.dual_sampler_type == 'uniform': # with replacement
                sample_index_b = random.randint(0, self.__len__() - 1)
            else:
                assert -1

            img_b, target_b = self.data[sample_index_b], self.targets[sample_index_b]
            img_b = Image.fromarray(img_b)

            if self.transform is not None:
                img_w_b = self.transform(img_b)
            if self.strong_transform is not None:
                img_s_b = self.strong_transform(img_b)
            if self.target_transform is not None:
                target_b = self.target_transform(target_b)

        else:
            img_w_b, img_s_b, target_b, sample_index_b = None, None, None, -1

        if self.transform is not None:
            img_w = self.transform(img)
        if self.strong_transform is not None:
            img_s = self.strong_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if img_w_b is None:
            img_w_b, img_s_b, target_b, sample_index_b = img_w, img_s, target, sample_index_a

        return [img_w, img_s], target, [img_w_b, img_s_b], target_b, sample_index_a, sample_index_b


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
    subset_thr = [100, 20]

# class IMBALANCECIFAR10_INFERENCE(torchvision.datasets.CIFAR10):
#     def __init__(self, root, args, train=True, transform=None, target_transform=None, download=False, train_dataset=None):
#         super(IMBALANCECIFAR10_INFERENCE, self).__init__(root)
#         self.args = args
#         self.data = train_dataset.data
#         self.targets = train_dataset.targets
#         self.transform = transform
#         self.target_transform = target_transform


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index from sampler
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]
        
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target, index

# class IMBALANCECIFAR100_INFERENCE(torchvision.datasets.CIFAR100):
#     def __init__(self, root, args, train=True, transform=None, target_transform=None, download=False, train_dataset=None):
#         super(IMBALANCECIFAR100_INFERENCE, self).__init__(root)
#         self.args = args
#         self.data = train_dataset.data
#         self.targets = train_dataset.targets
#         self.transform = transform
#         self.target_transform = target_transform

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index from sampler
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """

#         img, target = self.data[index], self.targets[index]
#         # to return a PIL Image
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return img, target, index



if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()