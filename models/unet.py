import torch 
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size = 3, padding = 'same', stride = 1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        c = self.batch_norm(x)
        c = self.conv1(c)
        c = self.relu(c)
        c = self.conv2(c)
        c = self.relu(c)
        p = self.max_pool(c)
        
        return c, p
    
class UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, kernel_size = 3, padding = 'same', stride = 1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_c)
        self.up_sample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_c + skip_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu = nn.ReLU()
        

    def forward(self, x, skip):
        c = self.batch_norm(x)
        us = self.up_sample(c)
        concat = torch.cat([us, skip], axis=1)
        c = self.conv1(concat)
        c = self.relu(c)
        c = self.conv2(c)
        c = self.relu(c)

        return c
    

class BottleNeck(nn.Module):
    def __init__(self, in_c, out_c, kernel_size = 3, padding = 'same', stride = 1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size = kernel_size, stride = stride, padding = padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        c = self.batch_norm(x)
        c = self.conv1(c)
        c = self.relu(c)
        c = self.conv2(c)
        c = self.relu(c)

        return c

class Unet(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        size = [16, 32, 64, 128, 256]
        self.down_block1 = DownBlock(in_c, size[0])
        self.down_block2 = DownBlock(size[0], size[1])
        self.down_block3 = DownBlock(size[1], size[2])
        self.down_block4 = DownBlock(size[2], size[3])

        self.bottleneck = BottleNeck(size[3], size[4])

        self.up_block1 = UpBlock(size[4], size[3], size[3])
        self.up_block2 = UpBlock(size[3], size[2], size[2])
        self.up_block3 = UpBlock(size[2], size[1], size[1])
        self.up_block4 = UpBlock(size[1], size[0], size[0])

        self.head = nn.Conv2d(size[0], 1, kernel_size=1)

    def forward(self, x):
        c1, p1 = self.down_block1(x)
        c2, p2 = self.down_block2(p1)
        c3, p3 = self.down_block3(p2)
        c4, p4 = self.down_block4(p3)

        bn = self.bottleneck(p4)

        u1 = self.up_block1(bn, c4)
        u2 = self.up_block2(u1, c3)
        u3 = self.up_block3(u2, c2)
        u4 = self.up_block4(u3, c1)

        output = self.head(u4)

        return output
    


if __name__ == "__main__":
    x = torch.rand((5, 3, 480, 640))
    model = Unet(in_c = 3)
    pred = model(x)
    print(pred.shape)
