import torchvision
import torchvision.transforms as transforms
from pdb import set_trace as bp

class TorchDatasets():
	"""
	A common class to get all inbuilt torch datasets
	"""
	def __init__(self, datasetname, rootpath, trainflag, downloadflag, transforms=None):
		dataset_obj = getattr(torchvision.datasets, datasetname)
		dataset_to_return = dataset_obj(root=rootpath,
		                                train=trainflag,
		                                download=downloadflag,
		                                transform=transforms)
		self.dataset = dataset_to_return

