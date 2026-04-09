import torch
import torch.distributed as dist
from typing import List
from src.topology.mesh import VirtualDeviceMesh
from src.engine.logger import DistributedLogger


class DataParallelSynchronizer:
    """
    Manages Data Parallelism by synchronizing model gradients across the
    Data Parallel (DP) process group.
    """

    def __init__(self, mesh: VirtualDeviceMesh) -> None:
        """
        Initializes the DataParallelSynchronizer.

        Args:
            mesh: The VirtualDeviceMesh containing topology information.
        """
        self.mesh: VirtualDeviceMesh = mesh
        self.dp_group: dist.ProcessGroup = mesh.get_dp_group()
        self.logger: DistributedLogger = DistributedLogger(rank=mesh.rank)

    def synchronize_gradients(self, parameters: List[torch.Tensor]) -> None:
        """
        Averages gradients across all processes in the data parallel group using all_reduce.

        Args:
            parameters: A list of tensors (typically model.parameters()) to be synchronized.
        """
        dp_size: int = self.mesh.dp_size

        if dp_size <= 1:
            return

        self.logger.debug("Synchronizing gradients across DP group.")

        for param in parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                param.grad.data.div_(dp_size)

        self.logger.debug("Gradient synchronization complete.")
