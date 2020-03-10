import torch
from base import BaseDataLoader

class CIFAR10Loader(BaseDataLoader):
	def __init__(self, dataset, batch_size, shuffle=False, num_workers=0):
		super().__init__(dataset, batch_size, shuffle, num_workers)

