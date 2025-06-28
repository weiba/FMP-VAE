import torch
import numpy as np
import scipy.io as sio

class SetDataset:
    def __init__(self,data,batch_size):
        self.cl_list = data[0]
        self.meta_data = data[1]
        self.sub_meta_idx = data[2]
        self.Methy_data = data[3]
        self.Mirna_data = data[4]
        self.sub_dataloader = []
        sub_data_loader_params = dict(batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,  # use main thread only or may receive multiple batches
                                      pin_memory=False)
        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.meta_data, self.sub_meta_idx[cl], cl, self.Methy_data,  self.Mirna_data)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))


    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)


class SubDataset:
    def __init__(self, meta_data, sub_meta_idx, cl,  Methy_data, Mirna_data):
        self.meta_data = meta_data
        self.sub_meta_idx = sub_meta_idx
        self.cl = cl
        self.Methy = Methy_data
        self.Mirna = Mirna_data

    def __getitem__(self, i):
        lineno = self.sub_meta_idx[i]
        # features = lineno
        features = self.meta_data[lineno]
        Methy_data = self.Methy[lineno]
        Mirna_data = self.Mirna[lineno]
        target = self.cl
        # target = lineno

        return  features, lineno, Methy_data, Mirna_data

    def __len__(self):
        return len(self.sub_meta_idx)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
