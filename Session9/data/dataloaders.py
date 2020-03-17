import torch 
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders, support for custom sampler and collate function added
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, sampler=None, collate_fn=default_collate):

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'sampler':sampler,
            'collate_fn': collate_fn                      
        }

        super().__init__(**self.init_kwargs)