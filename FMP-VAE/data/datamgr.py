import torch
from data.dataset import SetDataset, EpisodicBatchSampler
from abc import abstractmethod


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SetDataManager(DataManager):
    def __init__(self, n_way, n_support, n_query, n_eposide,  num_workers=4):
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.n_eposide = n_eposide
        self.batch_size = n_support + n_query
        self.num_workers = num_workers

    def get_data_loader(self, data):  # parameters that would change on train/val set
        dataset = SetDataset(data, self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=self.num_workers, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


