import pytest
import os
import torch
import torch.multiprocessing as mp
from typing import List

from src.topology.mesh import VirtualDeviceMesh
from src.parallelism.data import DataParallelSynchronizer
from src.parallelism.tensor import TensorParallelLinear
from src.parallelism.pipeline import PipelineScheduler


def _worker_parallelism(rank: int, world_size: int) -> None:
    """Worker to test parallel modules."""
    dp_size = 2
    tp_size = 2
    pp_size = 2

    mesh = VirtualDeviceMesh(
        world_size=world_size,
        rank=rank,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size
    )

    dp_sync = DataParallelSynchronizer(mesh)

    param = torch.nn.Parameter(torch.ones(10))
    if mesh.dp_group is not None:
        dp_rank = torch.distributed.get_rank(mesh.dp_group)
        param.grad = torch.full_like(param, float(dp_rank * 2 + 1))

        dp_sync.synchronize_gradients([param])

        if mesh.dp_size > 1:
            assert torch.allclose(param.grad, torch.full_like(param, 2.0))

    tp_linear = TensorParallelLinear(mesh, 16, 16)

    assert tp_linear.weight.shape == (8, 16)

    x = torch.randn(4, 16)
    y = tp_linear(x)
    assert y.shape == (4, 16)

    pp_sched = PipelineScheduler(mesh, num_microbatches=2)
    inputs = [1, 2]

    calls: List[int] = []
    def dummy_forward_step(x: int) -> int:
        calls.append(x)
        return x

    pp_sched.execute(dummy_forward_step, inputs)

    assert len(calls) == 2

    mesh.cleanup()


def test_parallel_modules():
    """Test Data, Tensor, and Pipeline Parallelism modules."""
    world_size = 8
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"

    mp.spawn(
        _worker_parallelism,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
