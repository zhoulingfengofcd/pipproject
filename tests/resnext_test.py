from zlf_resnext import Resnext
import torch

net = Resnext(10)
x = torch.rand((10, 3, 224, 224))
for name, layer in net.named_children():
    if name != "fc":
        x = layer(x)
        print(name, 'output shaoe:', x.shape)
    else:
        x = x.view(x.size(0), -1)
        x = layer(x)
        print(name, 'output shaoe:', x.shape)