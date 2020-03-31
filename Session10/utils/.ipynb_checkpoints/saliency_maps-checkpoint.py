import cv2
import torch
import numpy as np
from pdb import set_trace as bp
import matplotlib.pyplot as plt

class Hook:
    def __init__(self,
                 module,
                 backward=False):      
        if backward==False:
            self.myhook = module.register_forward_hook(self.myforward_hook_fn)
        else:
            self.myhook = module.register_backward_hook(self.mybackward_hook_fn)

    def myforward_hook_fn(self, module, input, output):   
        self.input = input
        self.output = output.data.cpu()

    def mybackward_hook_fn(self, module, grad_input, grad_output):      
        self.grad_input = grad_input
        self.grad_output = grad_output[0].data.cpu()

    def close(self):
        self.myhook.remove()

class GradCam:
    ''' 
    A GradCAM module which works with any PyTorch model.
        Arguments:
        model: PyTorch model object
        target_layer_names: names of the layer(s) in a list, according to the model.named_modules()
    Upon calling this class, it prints the gradCAM visualisations if the corresponding argument is True, along with return the visualisable ndarrays.
    '''
    def __init__(self, model, target_layer_names, use_cuda, weights_path=None):      
        self.all_fmaps = {}
        self.all_grads = {}
        self.model = model
        self.target_layers = target_layer_names
        self.cuda = use_cuda
        if weights_path is not None:
            if self.cuda:
                self.model.load_state_dict(torch.load(weights_path)['state_dict'])
            else:
                self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
        if self.cuda:
            self.model = model.cuda()
        self._add_hooks()

    def _add_hooks(self):
        for name, module in self.model.named_modules():
            self.all_fmaps[name] = Hook(module)
            self.all_grads[name] = Hook(module, backward=True)  

    def _process_image_and_display_cam(self, 
                                        image_to_test,
                                        cam,
                                        dataset_mean, 
                                        dataset_std, 
                                        max_pixel=255.0, 
                                        display=True,
                                        columns=0,
                                        rows=0):
    
        b, c, h, w = image_to_test.size()
        
        # convert cam to 0-255
        cam = cam * max_pixel
        cam = cam[:,None,:,:]
        
        # upsample cam to the image size
        saliency_map = torch.nn.functional.upsample(cam, size=(h, w), mode='bilinear', align_corners=False)

        mean = torch.FloatTensor(dataset_mean).view(1, 3, 1, 1).expand_as(image_to_test).to(image_to_test.device)
        mean = mean * max_pixel
        std = torch.FloatTensor(dataset_std).view(1, 3, 1, 1).expand_as(image_to_test).to(image_to_test.device)
        std = std * max_pixel
        
        # convert images back to 0-255. See how normalize was done here
        # https://github.com/albumentations-team/albumentations/blob/4f35cda5adf73ab8ceacbba827400a1348cfb77f/albumentations/augmentations/functional.py#L131
        denormalized_image = image_to_test.mul(std).add(mean)

        display_images = []
        
        fig = plt.figure(figsize=(12,12))
        for i in range(1, columns*rows+1):
            eachimage_numpy = (denormalized_image[i-1]).type(torch.IntTensor).permute(1,2,0).numpy().astype('uint8')
            eachimage_gcam = cv2.applyColorMap(np.uint8(saliency_map[i-1].numpy().transpose(1, 2, 0)), cv2.COLORMAP_JET)
            cam_on_image = cv2.addWeighted(eachimage_numpy,0.75,eachimage_gcam,0.25,0)    
            if(cam_on_image.max() != 0):     
                cam_on_image = cam_on_image / cam_on_image.max() * max_pixel
            cam_on_image = cv2.resize(cam_on_image, (200, 200))
            if display:
                fig.add_subplot(rows, columns, i)
                plt.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                plt.subplots_adjust(wspace=0.1, hspace=0.01, top=0.5 ,bottom=0)
                plt.imshow(cv2.cvtColor(cam_on_image.astype('uint8'), cv2.COLOR_BGR2RGB), aspect='equal')
            display_images.append(cam_on_image)
        return display_images                 

    def __call__(self,
                 image_to_test,
                 dataset_mean,
                 dataset_std,
                 transform=None,
                 index = None,
                 display=True,
                 columns=0,
                 rows=0):

        if transform is not None:
            # This should be used when passing unprocessed raw images. 
            # Remember transform expects obj in HWC
            image_to_test = transform(image_to_test).permute(1,0,2,3)
    
        if self.cuda:
            output = self.model(image_to_test.cuda())
        else:
            output = self.model(image_to_test)

        if index == None:
            index = torch.from_numpy(np.argmax(output.cpu().data.numpy(), axis=1))

        one_hot = torch.zeros_like(output, dtype=torch.float32)
        
        one_hot[np.arange(one_hot.shape[0]), index] = 1
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        else:
            one_hot = torch.sum(one_hot * output, dim=1)

        self.model.zero_grad()
        one_hot.backward(torch.ones_like(one_hot), retain_graph=True)

        output_gcams = {}
        for each_target_layer in self.target_layers:
            targetlayer_gradient = self.all_grads[each_target_layer].grad_output
            targetlayer_output = self.all_fmaps[each_target_layer].output
            map_size = targetlayer_gradient.size()[2:]
            channelwise_mean_grad = torch.nn.AvgPool2d(map_size)(targetlayer_gradient)
            
            gcam_output = (targetlayer_output * channelwise_mean_grad.expand_as(targetlayer_output))
            gcam_output = torch.sum(gcam_output, dim=1)
            gcam_output = torch.nn.ReLU()(gcam_output)
            
            # min-max norm on each image's gcam in the batch
            gcam_output_mins, _ = torch.min(gcam_output.view(gcam_output.shape[0], -1), dim=1)
            gcam_output_maxs, _ = torch.max(gcam_output.view(gcam_output.shape[0], -1), dim=1)
            gcam_output = gcam_output - gcam_output_mins[:, None, None]
            gcam_output = gcam_output / gcam_output_maxs[:, None, None]

            display_images = self._process_image_and_display_cam(image_to_test,
                                                                gcam_output,
                                                                dataset_mean,
                                                                dataset_std,
                                                                display=display,
                                                                columns=columns,
                                                                rows=rows)
            output_gcams[each_target_layer] = display_images

        return output_gcams

