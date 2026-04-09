import os
import torch.multiprocessing as mp
from src.topology.mesh import VirtualDeviceMesh
from src.engine.trainer import SimulatedTrainer
from src.engine.logger import DistributedLogger


def worker(
    rank: int,
    world_size: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    num_steps: int
) -> None:
    """
    Worker function executed by each virtual process.

    Args:
        rank: The global rank of this process.
        world_size: The total number of processes.
        dp_size: Number of processes in the Data Parallel group.
        tp_size: Number of processes in the Tensor Parallel group.
        pp_size: Number of processes in the Pipeline Parallel group.
        num_steps: The number of simulated training steps to execute.
    """
    mesh = VirtualDeviceMesh(
        world_size=world_size,
        rank=rank,
        dp_size=dp_size,
        tp_size=tp_size,
        pp_size=pp_size
    )

    trainer = SimulatedTrainer(
        mesh=mesh,
        hidden_size=256,
        num_microbatches=4
    )

    for step in range(num_steps):
        logger = DistributedLogger(rank=rank)
        logger.info(f"--- Iteration {step + 1} ---")
        trainer.train_step()

    mesh.cleanup()


def main() -> None:
    """
    Entry point for the distrib-3d-sim program.

    Sets up the configuration for 3D parallelism and spawns the virtual
    processes using torch.multiprocessing.
    """
    dp_size: int = 2
    tp_size: int = 2
    pp_size: int = 2
    world_size: int = dp_size * tp_size * pp_size
    num_steps: int = 2

    print(f"Starting 3D Parallel Simulation with {world_size} processes.")
    print(f"Configuration: DP={dp_size}, TP={tp_size}, PP={pp_size}")

    os.environ["OMP_NUM_THREADS"] = "1"

    mp.spawn(
        worker,
        args=(world_size, dp_size, tp_size, pp_size, num_steps),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
