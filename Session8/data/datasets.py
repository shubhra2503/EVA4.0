import torchvision
import torchvision.transforms as transforms


class TorchDatasets():
	"""
	A common class to get all inbuilt torch datasets
	"""
	def __init__(self, datasetname, rootpath, trainflag, downloadflag, transforms):
		dataset_to_return = getattr(torchvision.datasets, datasetname)
		setattr(dataset_to_return, 'root', rootpath)
		setattr(dataset_to_return, 'train', trainflag)
		setattr(dataset_to_return, 'download', downloadflag)
		setattr(dataset_to_return, 'transform', transforms)
		return dataset_to_return

