import torch as pt
import torch.nn.functional as F
import axon




class Model:
    def __init__(self, device):
        self.layer_1 = axon.conv2D(1, 8, device=device)
        self.layer_2 = axon.conv2D(8, 16, device=device)
        self.layer_3 = axon.MLP([64, 4], 2048, device=device)

    def parameters(self):
        P = []
        P.extend(self.layer_1.parameters())
        P.extend(self.layer_2.parameters())
        P.extend(self.layer_3.parameters())
        return P

    def __call__(self, img):
        r1 = self.layer_1(img)
        r1 = pt.relu(r1)
        r1 = F.max_pool2d(r1, 2)

        r2 = self.layer_2(r1)
        r2 = pt.relu(r2)
        r2 = F.max_pool2d(r2, 2)

        y = r2.reshape(r2.shape[0], -1)

        r3 = self.layer_3(y)

        return r3