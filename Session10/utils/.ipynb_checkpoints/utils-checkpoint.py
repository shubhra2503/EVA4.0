from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_summary(model, input):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = model.to(device)
	return summary(model, input_size=input)

def imshow(img, scale=0.5, shift=0.5):
    img = img*scale + shift     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def imshow_cv2(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def imshow_nd(img, scale=1, shift=0):
    img = img*scale + shift
    plt.imshow(img)

def get_datasetmean_std(dataset):
    # x = np.concatenate([np.asarray(dataset[i][0]) for i in range(len(dataset))])
    # train_mean = np.mean(x, axis=(0, 1))
    # train_std = np.std(x, axis=(0, 1))
    mean = np.mean(dataset.data, axis=(0,1,2))/255
    std = np.std(dataset.data, axis=(0,1,2))/255
    return mean, std
