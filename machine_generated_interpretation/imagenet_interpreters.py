import os
import cv2
import sys
import json
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.models import vgg16, vgg19
from torchvision import models

import torchray
# please install torchray package first
# refer Torchray package at: https://github.com/facebookresearch/TorchRay 
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.attribution.gradient import gradient


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x



class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x



class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam



def resize_postfn(grad, scale=4):
    """
    input size:
    grad: [img_size, img_size]
    To resize map: [3, 224, 224] -> [112, 112]
    """
    grad = grad.unsqueeze(0) if len(
        grad.size()) == 3 else grad   # ensure len(grad.size()) == 4
    grad = grad.abs().max(1, keepdim=True)[0] if grad.size(1) == 3 else grad     # grad shape = [1, 3, 32, 32]
    grad = F.avg_pool2d(grad, scale).squeeze(1)        # grad shape = [1, 1, 32, 32]
    shape = grad.shape                             # grad shape = [1, 8, 8]
    grad = grad.view(len(grad), -1)
    grad_min = grad.min(1, keepdim=True)[0]
    grad = grad - grad_min
    grad_max = grad.max(1, keepdim=True)[0]
    grad = grad / torch.max(grad_max, torch.tensor([1e-8], device=device))
    return grad.view(*shape)



def _norm(img):
    lim = [img.min(), img.max()]
    img = img - lim[0]  # also makes a copy
    img.mul_(1 / (lim[1] - lim[0]))
    img = torch.clamp(img, min=0, max=1)
    return img



def generate_extremal_perturbation(model, x, y, area=0.1):
    masks, cam = extremal_perturbation(
        model, x, y,
        reward_func=contrastive_reward,
        debug=True,
        areas=[area],
    )
    return cam[0]



def generate_gradcam(model, x, y):

    img = _norm(x[0]).detach().cpu().numpy().transpose(1,2,0)
    x = _norm(x)

    grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=True)
    mask = grad_cam(x.requires_grad_(True), y)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam



def generate_gradsaliency(model, x, y):

    img = _norm(x[0]).detach().cpu().numpy().transpose(1,2,0)
    x = _norm(x)

    mask = gradient(model, x, y)
    mask = resize_postfn(mask, scale=4).detach().cpu().numpy()   # [56, 56]
    mask = cv2.resize(mask[0], (224, 224), interpolation=cv2.INTER_LINEAR)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam
