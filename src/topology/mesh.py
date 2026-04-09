import os
import torch.distributed as dist
from typing import Dict, List, Optional
from src.engine.logger import DistributedLogger


class VirtualDeviceMesh:
    """
    Manages the virtual topology of nodes and GPUs for the distributed simulation.

    Treats local CPU processes as independent devices. Establishes torch.distributed
    process groups for Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline
    Parallelism (PP) using the gloo backend.
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        dp_size: int,
        tp_size: int,
        pp_size: int
    ) -> None:
        """
        Initializes the VirtualDeviceMesh and configures the distributed environment.

        Args:
            world_size: Total number of processes in the distributed run.
            rank: The rank of the current process.
            dp_size: Number of processes in the data parallel group.
            tp_size: Number of processes in the tensor parallel group.
            pp_size: Number of processes in the pipeline parallel group.

        Raises:
            ValueError: If the product of dp_size, tp_size, and pp_size does not equal world_size.
        """
        if dp_size * tp_size * pp_size != world_size:
            raise ValueError(
                "The product of DP, TP, and PP sizes must equal the total world_size."
            )

        self.world_size: int = world_size
        self.rank: int = rank
        self.dp_size: int = dp_size
        self.tp_size: int = tp_size
        self.pp_size: int = pp_size

        self.logger: DistributedLogger = DistributedLogger(rank=rank)

        self._init_distributed()

        self.dp_group: Optional[dist.ProcessGroup] = None
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.pp_group: Optional[dist.ProcessGroup] = None

        self._create_process_groups()

    def _init_distributed(self) -> None:
        """
        Initializes the default torch.distributed process group using the gloo backend.
        """
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size
        )
        self.logger.info("Initialized default process group using gloo backend.")

    def _create_process_groups(self) -> None:
        """
        Creates sub-process groups for Data, Tensor, and Pipeline parallelism based
        on the world_size and specified dimensions.

        This topology maps a 1D sequence of ranks into a 3D coordinate system (dp, pp, tp).
        """
        num_tp_groups: int = self.world_size // self.tp_size
        for i in range(num_tp_groups):
            ranks: List[int] = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group: dist.ProcessGroup = dist.new_group(ranks=ranks)
            if self.rank in ranks:
                self.tp_group = group
                self.logger.debug(f"Assigned to TP group with ranks: {ranks}")

        num_pp_groups: int = self.world_size // self.pp_size
        for i in range(self.dp_size):
            for j in range(self.tp_size):
                ranks: List[int] = [
                    (i * self.pp_size * self.tp_size) + (k * self.tp_size) + j
                    for k in range(self.pp_size)
                ]
                group: dist.ProcessGroup = dist.new_group(ranks=ranks)
                if self.rank in ranks:
                    self.pp_group = group
                    self.logger.debug(f"Assigned to PP group with ranks: {ranks}")

        num_dp_groups: int = self.world_size // self.dp_size
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                ranks: List[int] = [
                    (k * self.pp_size * self.tp_size) + (i * self.tp_size) + j
                    for k in range(self.dp_size)
                ]
                group: dist.ProcessGroup = dist.new_group(ranks=ranks)
                if self.rank in ranks:
                    self.dp_group = group
                    self.logger.debug(f"Assigned to DP group with ranks: {ranks}")

    def get_dp_group(self) -> dist.ProcessGroup:
        """
        Returns the data parallel process group.

        Returns:
            The Data Parallel ProcessGroup.
        """
        if self.dp_group is None:
            raise RuntimeError("DP group is not initialized.")
        return self.dp_group

    def get_tp_group(self) -> dist.ProcessGroup:
        """
        Returns the tensor parallel process group.

        Returns:
            The Tensor Parallel ProcessGroup.
        """
        if self.tp_group is None:
            raise RuntimeError("TP group is not initialized.")
        return self.tp_group

    def get_pp_group(self) -> dist.ProcessGroup:
        """
        Returns the pipeline parallel process group.

        Returns:
            The Pipeline Parallel ProcessGroup.
        """
        if self.pp_group is None:
            raise RuntimeError("PP group is not initialized.")
        return self.pp_group

    def cleanup(self) -> None:
        """
        Cleans up the distributed environment and destroys process groups.
        """
        dist.destroy_process_group()
        self.logger.info("Destroyed distributed process group.")
