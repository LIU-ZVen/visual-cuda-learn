import pytest
import torch
from vcl import vcl_C


@pytest.mark.parametrize("numel", [4, 5])
def test_relu_fp32(numel):
    x = torch.randn(numel, device="cuda:0")
    golden = torch.relu(x)
    output = torch.ops.vcl_C.relu(x)
    # print(f"golden: {golden}")
    # print(f"output: {output}")
    torch.testing.assert_close(output, golden)


@pytest.mark.parametrize("numel", [4, 5])
def test_relu_fp16(numel):
    x = torch.randn(numel).cuda().half().contiguous()
    golden = torch.relu(x)
    output = torch.ops.vcl_C.relu(x)
    print(f"\nx (fp16): {x}")
    print(f"golden (fp16): {golden}")
    print(f"output (fp16): {output}")
    torch.testing.assert_close(output, golden, atol=1e-3, rtol=1e-3)
