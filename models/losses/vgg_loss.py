import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import transforms, models

class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # breakpoint()
    def get_features(self, model, x):
        features = []
        target_layers = ['7', '17', '14', '21', '24']
        # target_layers = ['4', '9', '16', '23', '30']
        for name, layer in model.features._modules.items():
            # breakpoint()
            x = layer(x)
            if name in target_layers:
                features.append(x)
        return features

    def forward(self, x):
        return self.get_features(self.vgg, x)
        
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        # breakpoint()
        self.vgg.load_state_dict(torch.load("/home/dangpb1/Research/CamSpecDeblurring/uvcgan2/real_world_deblurring/advanced_recon/vgg_model.pth"))
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(5):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(5, 10):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(10, 17):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(17, 24):
        #     self.slice4.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(24, 31):
        #     self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGG16(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        self.vgg = VGGModel()
        # breakpoint()
        # self.vgg.load_state_dict(torch.load("/home/dangpb1/Research/CamSpecDeblurring/uvcgan2/real_world_deblurring/advanced_recon/vgg16_model_l1mean_best.pth"))
        vgg_pretrained_features = self.vgg.vgg.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # '7', '17', '14', '21', '24'
        for x in range(8):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 15):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(15, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 22):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 25):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG16().cuda()
        self.criterion = nn.L1Loss(reduction='mean')
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.weights = [1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0 / 1, 1.0]

    def forward(self, x, y):
        # breakpoint()
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # breakpoint()
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# class VGGPerceptualLoss(nn.Module):
#     def __init__(self, layer_names=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']):
#         super(VGGPerceptualLoss, self).__init__()
#         self.vgg_model = models.vgg19(pretrained=True).features
#         self.layer_names = layer_names
#         self.layers = [None] * len(layer_names)
#         for idx, (name, module) in enumerate(self.vgg_model.named_children()):
#             if f'conv{idx+1}_2' in layer_names:
#                 self.layers[layer_names.index(f'conv{idx+1}_2')] = module
#             if all(layer is not None for layer in self.layers):
#                 break
#         for layer in self.layers:
#             if layer is None:
#                 raise ValueError(f"At least one of the layers in {layer_names} does not exist in the VGG19 model.")

#         # Freeze the parameters of the VGG model
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#         for param in self.vgg_model.parameters():
#             param.requires_grad = False

#     def forward(self, input_image, target_image):
#         input_features = [layer(input_image) for layer in self.layers]
#         target_features = [layer(target_image) for layer in self.layers]
#         loss = 0
#         for input_feature, target_feature in zip(input_features, target_features):
#             loss += torch.mean((input_feature - target_feature) ** 2)
#         return loss

# Example usage:
# Assuming 'generated_image' and 'target_image' are torch tensors with the same size
# loss_fn = VGGPerceptualLoss(layer_names=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3'])
# loss = loss_fn(generated_image, target_image)
