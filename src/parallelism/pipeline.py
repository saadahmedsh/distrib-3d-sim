import torch
import torch.distributed as dist
from typing import Callable, Any, List
from src.topology.mesh import VirtualDeviceMesh
from src.engine.logger import DistributedLogger


class PipelineScheduler:
    """
    Implements a simple micro-batching scheduler for Pipeline Parallelism.

    Coordinates the execution of micro-batches across pipeline stages to minimize
    idle bubbles. In this simulation, it supports a fill-and-drain schedule for
    a two-stage pipeline.
    """

    def __init__(self, mesh: VirtualDeviceMesh, num_microbatches: int) -> None:
        """
        Initializes the PipelineScheduler.

        Args:
            mesh: The VirtualDeviceMesh containing topology information.
            num_microbatches: The number of micro-batches to split the global batch into.
        """
        self.mesh: VirtualDeviceMesh = mesh
        self.pp_group: dist.ProcessGroup = mesh.get_pp_group()
        self.pp_rank: int = dist.get_rank(self.pp_group)
        self.pp_size: int = mesh.pp_size
        self.num_microbatches: int = num_microbatches
        self.logger: DistributedLogger = DistributedLogger(rank=mesh.rank)

        if self.pp_size != 2:
            self.logger.info(
                f"Warning: PipelineScheduler is designed for a 2-stage pipeline. "
                f"Current pp_size is {self.pp_size}."
            )

    def execute(self, forward_step_fn: Callable[..., Any], inputs: List[Any]) -> None:
        """
        Executes the pipeline schedule over the provided micro-batches.

        Args:
            forward_step_fn: A function that executes a single forward/backward pass for a micro-batch.
            inputs: A list of inputs, one for each micro-batch.

        Raises:
            ValueError: If the number of inputs does not match num_microbatches.
        """
        if len(inputs) != self.num_microbatches:
            raise ValueError(
                f"Expected {self.num_microbatches} inputs, got {len(inputs)}."
            )

        self.logger.debug(f"Starting pipeline execution for {self.num_microbatches} micro-batches.")

        if self.pp_size == 1:
            for i in range(self.num_microbatches):
                forward_step_fn(inputs[i])
            return

        for step in range(self.num_microbatches + self.pp_size - 1):
            if self.pp_rank == 0:
                if step < self.num_microbatches:
                    self.logger.debug(f"Stage 0: Processing micro-batch {step}")
                    _ = forward_step_fn(inputs[step])
                    self.logger.debug(f"Stage 0: Sending activations to Stage 1 for micro-batch {step}")

            elif self.pp_rank == 1:
                if step >= 1 and (step - 1) < self.num_microbatches:
                    mb_idx = step - 1
                    self.logger.debug(f"Stage 1: Receiving activations from Stage 0 for micro-batch {mb_idx}")
                    self.logger.debug(f"Stage 1: Processing micro-batch {mb_idx}")
                    _ = forward_step_fn(inputs[mb_idx])
                    self.logger.debug(f"Stage 1: Sending gradients to Stage 0 for micro-batch {mb_idx}")

        self.logger.debug("Pipeline execution complete.")
