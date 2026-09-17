import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import pytest


@pytest.fixture(autouse=True)
def cpu_only():
    torch.set_num_threads(2)
    yield
    assert not torch.cuda.is_initialized(), "CPU tests must not initialize CUDA"
