"""

Multiview classifier:

1. Use CNN models to extract feature map
2. Concatenate 7 feature maps and predict four classes

"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from model_zoo import *
# from torchsummary import summary

class MVClassifier_Proposed(nn.Module):
    def __init__(self, model, numclasses,threshold=None):
        super(MVClassifier_Proposed, self).__init__()
        self.model = model
        self.threshold = threshold
        self.numclasses = numclasses

        # list model names from model_zoo.py
        model_channels = {
            'SqueezeNet1_0': 512,
            'SqueezeNet1_1': 512,
            'MobileNetV1': 1024,
            'MobileNetV2': 320,
            'MobileNetV3': 576,
            'ShuffleNetV2_x0_5': 192,
            'ShuffleNetV2_x1_0': 464,
            'EfficientNetV1': 1280,
        }
        channel = model_channels[model.__class__.__name__]

        self.SENet = SENet(channel*12)  # Adjusted for additional channels from autoencoder
        self.reduce = nn.Conv2d(in_channels=channel*12, out_channels=channel, kernel_size=3, padding=1)
        self.Pool = nn.AdaptiveAvgPool2d(1)
        self.fclayer = nn.Sequential(
            nn.Linear(channel, 128),
            nn.ReLU(),
            nn.Linear(128, self.numclasses)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0),  # 512x1x1 -> 256x3x3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 256x3x3 -> 128x6x6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 128x6x6 -> 64x12x12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x12x12 -> 32x24x24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 24, kernel_size=5, stride=1, padding=1),  # 32x24x24 -> 12x28x28
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0),  # 512x1x1 -> 256x3x3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 256x3x3 -> 128x6x6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 128x6x6 -> 64x12x12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x12x12 -> 32x24x24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 6, kernel_size=5, stride=1, padding=1),  # 32x24x24 -> 12x28x28
            nn.ReLU(inplace=True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=1),  # 4x100x100 -> 32x100x100
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x100x100 -> 64x50x50
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x50x50 -> 128x25x25
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128x25x25 -> 256x12x12
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),  # 256x12x12 -> 512x6x6
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, views, numclasses):
        fuse, logits = [], []

        x1 = torch.cat((views[3], views[8]), dim=1)  # dim=0은 채널 차원을 의미합니다.

        # 나머지 이미지를 선택하여 채널 차원에서 합칩니다.
        x2_images = [views[i] for i in range(10) if i not in [3, 8]]
        x2 = torch.cat(x2_images, dim=1)
        # print(x1.shape)
        # print(x2.shape)

        for view in views:
            output = self.model(view)
            logit = self.Pool(output).squeeze()
            logit = self.fclayer(logit) # End: (batch_size, output_class)
            logits.append(logit) # (views, batch_size, 4)
            fuse.append(output)

        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2)

        # print(encoded2.shape)
        decoded_feature1 = self.decoder1(encoded1)
        decoded_feature2 = self.decoder2(encoded2)

        #print(decoded_feature1.shape)
        # print(decoded_feature2.shape)
        fuse.append(encoded1)
        fuse.append(encoded2)

        fusion = torch.cat(fuse, dim=1)  # (batch_size, (512*7 + 16), 5, 5)
        fusion = self.SENet(fusion)
        fusion = self.reduce(fusion)  # (batch_size, 512, 5, 5)
        flatten = self.Pool(fusion).squeeze()  # (batch_size, 512)
        fusion_logit = self.fclayer(flatten)  # End: (batch_size, output_class)
        for batch in fusion_logit:
            if self.threshold:
                if F.softmax(batch, dim=0)[3] < self.threshold:
                    batch[3] = (F.softmax(batch, dim=0)).min()
        return logits, fusion_logit,x1, x2, decoded_feature1, decoded_feature2, encoded1, encoded2
class MVClassifier(nn.Module):
    def __init__(self, model, num_classes ,threshold=None):
        super(MVClassifier, self).__init__()
        self.model = model
        self.threshold = threshold
        self.num_classes = num_classes

        # list model names from model_zoo.py
        model_channels = {
            'SqueezeNet1_0': 512,
            'SqueezeNet1_1': 512,
            'MobileNetV1': 1024,
            'MobileNetV2': 320,
            'MobileNetV3': 576,
            'ShuffleNetV2_x0_5': 192,
            'ShuffleNetV2_x1_0': 464,
            'EfficientNetV1': 1280,
        }
        channel = model_channels[model.__class__.__name__]

        self.SENet = SENet(channel*10)  # Adjusted for additional channels from autoencoder
        self.reduce = nn.Conv2d(in_channels=channel*10, out_channels=channel, kernel_size=3, padding=1)
        self.Pool = nn.AdaptiveAvgPool2d(1)
        self.fclayer = nn.Sequential(
            nn.Linear(channel, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )


    def forward(self, views, combined_image):
        fuse, logits = [], []

        for view in views:
            output = self.model(view)
            logit = self.Pool(output).squeeze()
            logit = self.fclayer(logit) # End: (batch_size, output_class)
            logits.append(logit) # (views, batch_size, 4)
            fuse.append(output)
        fusion = torch.cat(fuse, dim=1)  # (batch_size, (512*7 + 16), 5, 5)
        fusion = self.SENet(fusion)
        fusion = self.reduce(fusion)  # (batch_size, 512, 5, 5)
        flatten = self.Pool(fusion).squeeze()  # (batch_size, 512)
        fusion_logit = self.fclayer(flatten)  # End: (batch_size, output_class)
        for batch in fusion_logit:
            if self.threshold:
                if F.softmax(batch, dim=0)[3] < self.threshold:
                    batch[3] = (F.softmax(batch, dim=0)).min()
        return logits, fusion_logit

class MVClassifier_Proposed_5(nn.Module):
    def __init__(self, model, numclasses,threshold=None):
        super(MVClassifier_Proposed, self).__init__()
        self.model = model
        self.threshold = threshold
        self.numclasses = numclasses

        # list model names from model_zoo.py
        model_channels = {
            'SqueezeNet1_0': 512,
            'SqueezeNet1_1': 512,
            'MobileNetV1': 1024,
            'MobileNetV2': 320,
            'MobileNetV3': 576,
            'ShuffleNetV2_x0_5': 192,
            'ShuffleNetV2_x1_0': 464,
            'EfficientNetV1': 1280,
        }
        channel = model_channels[model.__class__.__name__]

        self.SENet = SENet(channel*12)  # Adjusted for additional channels from autoencoder
        self.reduce = nn.Conv2d(in_channels=channel*12, out_channels=channel, kernel_size=3, padding=1)
        self.Pool = nn.AdaptiveAvgPool2d(1)
        self.fclayer = nn.Sequential(
            nn.Linear(channel, 128),
            nn.ReLU(),
            nn.Linear(128, self.numclasses)
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0),  # 512x1x1 -> 256x3x3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 256x3x3 -> 128x6x6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 128x6x6 -> 64x12x12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x12x12 -> 32x24x24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 15, kernel_size=5, stride=1, padding=1),  # 32x24x24 -> 12x28x28
            nn.ReLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0),  # 512x1x1 -> 256x3x3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),  # 256x3x3 -> 128x6x6
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  # 128x6x6 -> 64x12x12
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x12x12 -> 32x24x24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 15, kernel_size=5, stride=1, padding=1),  # 32x24x24 -> 12x28x28
            nn.ReLU(inplace=True)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=5, stride=1, padding=1),  # 4x100x100 -> 32x100x100
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x100x100 -> 64x50x50
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x50x50 -> 128x25x25
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128x25x25 -> 256x12x12
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),  # 256x12x12 -> 512x6x6
            nn.ReLU(inplace=True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, views, numclasses):
        fuse, logits = [], []

        x1 = torch.cat((views[3], views[8]), dim=1)  # dim=0은 채널 차원을 의미합니다.

        # 나머지 이미지를 선택하여 채널 차원에서 합칩니다.
        x2_images = [views[i] for i in range(10) if i not in [3, 8]]
        x2 = torch.cat(x2_images, dim=1)
        # print(x1.shape)
        # print(x2.shape)

        for view in views:
            output = self.model(view)
            logit = self.Pool(output).squeeze()
            logit = self.fclayer(logit) # End: (batch_size, output_class)
            logits.append(logit) # (views, batch_size, 4)
            fuse.append(output)

        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2)

        # print(encoded2.shape)
        decoded_feature1 = self.decoder1(encoded1)
        decoded_feature2 = self.decoder2(encoded2)

        #print(decoded_feature1.shape)
        # print(decoded_feature2.shape)
        fuse.append(encoded1)
        fuse.append(encoded2)

        fusion = torch.cat(fuse, dim=1)  # (batch_size, (512*7 + 16), 5, 5)
        fusion = self.SENet(fusion)
        fusion = self.reduce(fusion)  # (batch_size, 512, 5, 5)
        flatten = self.Pool(fusion).squeeze()  # (batch_size, 512)
        fusion_logit = self.fclayer(flatten)  # End: (batch_size, output_class)
        for batch in fusion_logit:
            if self.threshold:
                if F.softmax(batch, dim=0)[3] < self.threshold:
                    batch[3] = (F.softmax(batch, dim=0)).min()
        return logits, fusion_logit,x1, x2, decoded_feature1, decoded_feature2, encoded1, encoded2