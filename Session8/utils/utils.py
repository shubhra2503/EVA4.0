from torchsummary import summary

def get_summary(model, input):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = model.to(device)
	return summary(model, input_size=(3,32,32))