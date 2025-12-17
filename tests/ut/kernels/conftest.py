import pytest
import torch

@pytest.fixture(autouse=True, scope="module")
def torch_manual_seed():
    torch.manual_seed(3407)
