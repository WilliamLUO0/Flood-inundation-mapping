
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock,self).__init__()
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.prelu = nn.PReLU()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        input = x

        x1 = self.BN1(x)
        x1 = F.relu(x1)
        x1 = self.conv1(x1)

        add1_1 = self.conv1x1_1(input)
        x2 = x1 + add1_1

        x2 = self.BN2(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)

        add2_1 = self.conv1x1_2(input)

        x3 = x1 + x2 + add2_1

        return x3

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleLayer,self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.prelu = nn.PReLU()

    def forward(self, x, skip_connection):
        x = self.BN(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        return x

class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1x1,self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.BN(x)
        x = F.relu(x)
        x = self.conv(x)
        return x
    
class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def forward(self, x, mask):
        mask = mask.unsqueeze(1)
        x = x * mask
        x = torch.clamp(x, min=0)
        return x
        
class DenseUnet(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseUnet,self).__init__()
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
        
        self.dense1 = DenseBlock(in_channels=growth_rate, out_channels= growth_rate*2)
        self.dense2 = DenseBlock(in_channels=growth_rate*2, out_channels=growth_rate*4)
        self.dense3 = DenseBlock(in_channels=growth_rate*4, out_channels= growth_rate*8)
        self.dense4 = DenseBlock(in_channels=growth_rate*8, out_channels= growth_rate*16)
    
        self.upsample1 = UpsampleLayer(in_channels=growth_rate*16, out_channels=growth_rate*8)
        self.dense5 = DenseBlock(in_channels=growth_rate*8*2, out_channels=growth_rate*8)
    
        self.upsample2 = UpsampleLayer(in_channels=growth_rate*8, out_channels=growth_rate*4)
        self.dense6 = DenseBlock(in_channels=growth_rate*4*2, out_channels=growth_rate*4)
        
        self.upsample3 = UpsampleLayer(in_channels=growth_rate*4, out_channels=growth_rate*2)
        self.dense7 = DenseBlock(in_channels=growth_rate*2*2, out_channels=growth_rate*2)

        self.upsample4 = UpsampleLayer(in_channels=growth_rate*2, out_channels=growth_rate)
        
        self.BN3 = nn.BatchNorm2d(growth_rate*2)
        self.conv5 = nn.Conv2d(growth_rate*2, growth_rate*2, kernel_size=3, stride=1, padding=1)
        self.conv6 = conv1x1(in_channels=growth_rate*2, out_channels=1)

        self.mask = MaskLayer()
        self.prelu = nn.PReLU()

    def forward(self, x, mask):

        x = self.BN1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)

        x1 = self.dense1(x)
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2, padding=0)

        x2 = self.dense2(x1)
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2, padding=0)
    
        x3 = self.dense3(x2)
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2, padding=0)

        x4 = self.dense4(x3)
        x4 = F.max_pool2d(x4, kernel_size=2, stride=2, padding=0)

        x5 = self.upsample1(x4, x3)
        x5 = self.dense5(x5)

        x6 = self.upsample2(x5, x2)
        x6 = self.dense6(x6)

        x7 = self.upsample3(x6, x1)
        x7 = self.dense7(x7)

        x8 = self.upsample4(x7, x)
        x8 = self.BN3(x8)
        x8 = F.relu(x8)
        x8 = self.conv5(x8)

        x8 = self.conv6(x8)

        x8 = self.mask(x8, mask)
        return x8

if __name__ == '__main__':
    model1 = DenseUnet(in_channels=5, growth_rate=32)
    msk = torch.randn(1,128,128)
    input = torch.randn(1,5,128,128)

    output1 = model1(input, msk)
    print(output1.shape)
        






