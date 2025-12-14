from vcl import vcl_C
import torch

def hello_world():
    print("Hello, World!")


hello_world()

torch.ops.vcl_C.hello_gpu()

x = torch.randn(5, device="cuda:0")
y = torch.randn(5, device="cuda:0")

print(f"x: {x}")
print(f"y: {y}")

z = torch.ops.vcl_C.elementwise_add(x, y)
print(f"elementwise_add result: {z}")

relu = torch.ops.vcl_C.relu(x)
print(f"relu result: {relu}")

x_f16 = x.to(torch.float16)
print(f"x_f16: {x_f16}")
relu_f16 = torch.ops.vcl_C.relu(x_f16)
print(f"relu_f16 result: {relu_f16}")
