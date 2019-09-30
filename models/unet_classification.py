
import torch
from torch import nn
import torch.nn.functional as F

from models.unet.unet_model import UNet
from config import config

### Version_1 (not use inhertance) ###
# class UNet_classification(nn.Module):
#     def __init__(self, n_channels, n_classes):
#         super(UNet_classification, self).__init__()
#         opt = config
#         self.unet = UNet(3, 3)
#         self.unet.load_state_dict(torch.load(opt.load_model_path))
#
#     def forward(self, x):
#         # x = self.unet(x)
#         x1 = self.unet.inc(x)
#         x2 = self.unet.down1(x1)
#         x3 = self.unet.down2(x2)
#         x4 = self.unet.down3(x3)
#         x5 = self.unet.down4(x4)
#         x = self.unet.up1(x5, x4)
#         x = self.unet.up2(x, x3)
#         x = self.unet.up3(x, x2)
#         x = self.unet.up4(x, x1)
#         x = self.unet.outc(x)
#         return torch.sigmoid(x)

opt = config

### Version_2 (use inhertance) ###
class UNet_classification(UNet):
    def __init__(self, n_channels, n_classes):
        super(UNet_classification, self).__init__(n_channels=n_channels, n_classes=n_classes)
        self.load_state_dict(torch.load(opt.pretrain_model_path))

        ### freeze all layers ###
        for child in self.children():
            self.freeze_layer(child)

        ### freeze layer ###
        # self.freeze_layer(self.inc)
        # self.freeze_layer(self.down1)
        # self.freeze_layer(self.down2)
        # self.freeze_layer(self.down3)
        # self.freeze_layer(self.down4)
        # self.freeze_layer(self.up1)
        # self.freeze_layer(self.up2)
        # self.freeze_layer(self.up3)
        # self.freeze_layer(self.up4)
        # self.freeze_layer(self.outc)
        ### freeze layer ###

        test = list(enumerate(self.children()))
        print("Layers amount: {}".format(len(test)))

        ### Not yet upsampling, but after 4 downsamplings ###
        self.fc_input_size = int(512 * (opt.img_crop_height / (2 ** 4)) * (opt.img_crop_width / (2 ** 4)))

        ### After upsampling ###
        # self.fc_input_size = 64 * opt.img_crop_height * opt.img_crop_width

        print("FC_input_size_flatten: {}".format(self.fc_input_size))

        '''-----------------------------------------test---------------------------------------'''
        # self.fc1 = nn.Linear(self.fc_input_size, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)
        '''-----------------------------------------test---------------------------------------'''
        self.fc = nn.Linear(self.fc_input_size, opt.num_of_class)
        # self.fc = nn.Linear(64 * opt.img_crop_height * opt.img_crop_width, 2)

    def forward(self, x):
        # print("Origin: {}".format(x.shape))
        x1 = self.inc(x)
        x1 = nn.Dropout(opt.dr)(x1)
        # print("After inc: {}".format(x1.shape))
        x2 = self.down1(x1)
        x2 = nn.Dropout(opt.dr)(x2)
        # print("After down1: {}".format(x2.shape))
        x3 = self.down2(x2)
        x3 = nn.Dropout(opt.dr)(x3)
        # print("After down2: {}".format(x3.shape))
        x4 = self.down3(x3)
        x4 = nn.Dropout(opt.dr)(x4)
        # print("After down3: {}".format(x4.shape))
        x5 = self.down4(x4)
        x5 = nn.Dropout(opt.dr)(x5)
        # print("After down4: {}".format(x5.shape))

        ### Not yet upsampling ###
        glaucoma_predict = x5.view(x5.size(0), -1)
        glaucoma_predict = nn.Dropout(opt.dr)(glaucoma_predict)
        glaucoma_predict = self.fc(glaucoma_predict)
        # print("Glaucoma predict: {}".format(glaucoma_predict.shape))
        ### Not yet upsampling ###

        x = self.up1(x5, x4)
        # print("After up1: {}".format(x.shape))
        x = self.up2(x, x3)
        # print("After up2: {}".format(x.shape))
        x = self.up3(x, x2)
        # print("After up3: {}".format(x.shape))
        x = self.up4(x, x1)
        # print("After up4: {}".format(x.shape))

        ### After upsampling ###
        # glaucoma_predict = x.view(x.size(0), -1)
        # glaucoma_predict = x.view(-1, 64 * 256 * 256)

        ### avoid overfitting ###
        # glaucoma_predict = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # # print(glaucoma_predict.shape)
        # flatten_input_size = glaucoma_predict.size(1) * glaucoma_predict.size(2) * glaucoma_predict.size(3)
        # glaucoma_predict = glaucoma_predict.view(glaucoma_predict.size(0), -1)
        # glaucoma_predict = nn.Dropout(opt.dr)(glaucoma_predict)
        ### avoid overfitting ###

        # glaucoma_predict = self.fc(glaucoma_predict)
        # glaucoma_predict = nn.Linear(flatten_input_size, opt.num_of_class).cuda()(glaucoma_predict)
        # print("Glaucoma predict: {}".format(glaucoma_predict.shape))

        # glaucoma_predict = F.relu(self.fc1(glaucoma_predict))
        # glaucoma_predict = F.relu(self.fc2(glaucoma_predict))
        # glaucoma_predict = self.fc3(glaucoma_predict)
        ### After upsampling ###

        x = self.outc(x)
        # print("After outc: {}".format(x.shape))

        # return {"fake_image": torch.sigmoid(x), "prediction": torch.sigmoid(glaucoma_predict)}
        return {"fake_image": torch.sigmoid(x), "prediction": glaucoma_predict}


    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    model = UNet_classification(3, 3)
    print(model)

    opt = config
    print(opt)
