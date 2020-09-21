import torch
import torch.nn as nn


class EncoderBlock_(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(EncoderBlock_, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecoderBlock_(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock_, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

    def forward(self, x):
        return self.decode(x)


class unet(nn.Module):
    def __init__(self, num_classes):
        super(unet, self).__init__()
        self.enc1 = EncoderBlock_(3, 64)
        self.enc2 = EncoderBlock_(64, 128)
        self.enc3 = EncoderBlock_(128, 256)
        self.enc4 = EncoderBlock_(256, 256)

        self.dec4 = DecoderBlock_(256, 512)
        self.dec3 = DecoderBlock_(512 + 256, 256)
        self.dec2 = DecoderBlock_(256 + 128, 128)
        self.dec1 = DecoderBlock_(128 + 64, 64)

        self.dec0 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))
        dec0 = self.dec0(dec1)
        final = self.final(dec0)
        # nn.CrossEntropyLoss中内置了softmax
        # CrossEntropyLoss的input为没有softmax过的output, target为未经过one-hot的label
        return final
