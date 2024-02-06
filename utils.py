import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import importlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# WARNING:
# There is no guarantee that it will work or be used on a model. Please do use it with caution unless you make sure everything is working.
use_fp16 = False

if use_fp16:
    from torch.cuda.amp import autocast
else:
    class Autocast(): # This is a dummy autocast class
        def __init__(self):
            pass
        def __enter__(self, *args, **kwargs):
            pass
        def __call__(self, arg=None):
            if arg is None:
                return self
            return arg
        def __exit__(self, *args, **kwargs):
            pass

    autocast = Autocast()

def experiment_names(args, log_root, dataset, arch):
    sampler_list = ['default', 'balance', 'reverse']
    sampler_names = ['d', 'b', 'r']
    if args.loss_type == 'mixup':
        if args.dual_sample:
            dual_name = '_'.join([sampler_names[sampler_list.index(args.sampler_type)], sampler_names[sampler_list.index(args.dual_sampler_type)]])
        else:
            dual_name = 'conventional'
        dual_name += '_beta_' + str(args.mixup_beta)
        file_name = '_'.join([dataset, arch, args.loss_type, dual_name, args.train_rule, args.imb_type, str(args.imb_factor)])
        if args.temp_type:
            file_name += '_temp_' + str(args.temp_type) + '_' + str(args.temp_eta) + '_' + str(args.temp_epsilon)
    else:
        file_name = '_'.join([dataset, arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor)])
    # if args.cls_sample_method:
    #     file_name += '_cls_' + args.cls_sample_method
    # if args.ins_sample_method:
    #     file_name += '_ins_' + args.ins_sample_method
    if args.use_soft_weight and args.dual_sampler_type == 'reverse':
        file_name += '_SW_' + str(args.weight_gamma)

    if args.exp_str:
        log_root = log_root + '_' + args.exp_str
    return file_name, log_root

def make_dir(root):
    from torch.utils.tensorboard import SummaryWriter
    try:
        original_umask = os.umask(0)
        os.makedirs(root, exist_ok=True)
        writer = SummaryWriter(root)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(root):
            pass
        else:
            raise
    finally:
        os.umask(original_umask)

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.log_root, args.file_name, 'confusion_matrix.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_result(file_root, data, title, label=None, ylabel=None, xlabel=None, flag=None):
    if flag == 'per_cls':
        plt.plot(range(len(data)), data, label=label)
    elif flag == 'acc':
        plt.plot(range(len(data[0])), data[0], label=label[0])
        plt.plot(range(len(data[1])), data[1], label=label[1])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label is not None:
        plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(file_root, title + '.png'))
    plt.close()


def plot_context_result(file_root, data, title, label=None, ylabel=None, xlabel=None, flag=None):
    for i in range(len(label)):
        plt.plot(range(len(data)), data[label[i]])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label is not None:
        plt.legend(label)
    plt.grid(True)
    plt.savefig(os.path.join(file_root, title + '.png'))
    plt.close()

def save_checkpoint(file_root, state, is_best):
    
    filename = '%s/ckpt.pth.tar' % (file_root)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
        
def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def info(self, names=None):
        print(names)

    def warning(self, names=None):
        print("    *Warning :", names)

    def close(self):
        if self.file is not None:
            self.file.close()

def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_state_dict(model, state_dict, no_ignore=False):
    own_state = model.state_dict()
    count = 0
    for name, param in state_dict.items():
        if name not in own_state: # ignore
            print("Warning: {} ignored because it does not exist in state_dict".format(name))
            assert not no_ignore, "Ignoring param that does not exist in model's own state dict is not allowed."
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError as e:
            print("Error in copying parameter {}, source shape: {}, destination shape: {}".format(name, param.shape, own_state[name].shape))
            raise e
        count += 1
    if count != len(own_state):
        print("Warning: Model has {} parameters, copied {} from state dict".format(len(own_state), count))
    return count