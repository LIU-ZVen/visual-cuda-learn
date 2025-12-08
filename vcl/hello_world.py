def hello_world():
    print("Hello, World!")


from vcl import vcl_C
import torch

torch.ops.vcl_C.hello_gpu()
