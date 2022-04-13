# coding=utf-8
import numpy as np
import torch
import os


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, max_cls_samples, classes_per_it, num_samples, iterations, data_path):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations
        self.max_cls_samples = max_cls_samples
        self.classes, _ = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        class_names = []
        for (_, dirs, _) in os.walk(data_path):
            class_names.extend(dirs)
            break

        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, cname in enumerate(class_names):
            self.numel_per_class[idx] = len(os.listdir(os.path.join(data_path, cname)))

        # print("self.numel_per_class", self.numel_per_class)

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it
        batch = []
        for _ in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(c_idxs):
                s = slice(i * spc, (i + 1) * spc)
                sample_idxs = torch.randperm(self.numel_per_class[c])[:spc] # in tiny imagenet, all calsses have the same sample size
                sample_idxs += (sum(self.numel_per_class[:c-1]))
                batch[s] = sample_idxs
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
