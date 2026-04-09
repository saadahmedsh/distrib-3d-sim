import pytest
import os
import torch.multiprocessing as mp
from src.topology.mesh import VirtualDeviceMesh


def _worker_mesh(rank: int, world_size: int) -> None:
    """Worker to initialize mesh and test groups."""
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

    assert mesh.get_dp_group() is not None
    assert mesh.get_tp_group() is not None
    assert mesh.get_pp_group() is not None

    mesh.cleanup()


def test_virtual_device_mesh_initialization():
    """Test the initialization of the VirtualDeviceMesh."""
    world_size = 8
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    mp.spawn(
        _worker_mesh,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def test_invalid_mesh_configuration():
    """Test that a ValueError is raised when DP * TP * PP != world_size."""
    with pytest.raises(ValueError):
        VirtualDeviceMesh(
            world_size=8,
            rank=0,
            dp_size=2,
            tp_size=2,
            pp_size=3
        )
