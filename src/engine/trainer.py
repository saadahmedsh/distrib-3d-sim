import torch
import time
from typing import List
from src.topology.mesh import VirtualDeviceMesh
from src.parallelism.data import DataParallelSynchronizer
from src.parallelism.tensor import TensorParallelLinear
from src.parallelism.pipeline import PipelineScheduler
from src.engine.logger import DistributedLogger


class SimulatedTrainer:
    """
    Simulates a distributed training loop incorporating 3D Parallelism.

    Integrates the VirtualDeviceMesh, DataParallelSynchronizer, TensorParallelLinear,
    and PipelineScheduler to mimic the training process using synthetic data and time delays.
    """

    def __init__(self, mesh: VirtualDeviceMesh, hidden_size: int = 1024, num_microbatches: int = 4) -> None:
        """
        Initializes the SimulatedTrainer.

        Args:
            mesh: The VirtualDeviceMesh for distributed topology.
            hidden_size: The dimensionality of the linear layer features.
            num_microbatches: Number of micro-batches for pipeline parallelism.
        """
        self.mesh: VirtualDeviceMesh = mesh
        self.hidden_size: int = hidden_size
        self.num_microbatches: int = num_microbatches
        self.logger: DistributedLogger = DistributedLogger(rank=mesh.rank)

        self.model: TensorParallelLinear = TensorParallelLinear(
            mesh=mesh,
            in_features=hidden_size,
            out_features=hidden_size
        )
        self.dp_synchronizer: DataParallelSynchronizer = DataParallelSynchronizer(mesh=mesh)
        self.pipeline_scheduler: PipelineScheduler = PipelineScheduler(
            mesh=mesh,
            num_microbatches=num_microbatches
        )

        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def _mock_forward_backward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Simulates the forward and backward pass of a micro-batch.

        Args:
            input_tensor: A synthetic input tensor for the micro-batch.

        Returns:
            The output tensor from the forward pass.
        """
        time.sleep(0.1)

        output: torch.Tensor = self.model(input_tensor)

        target: torch.Tensor = torch.randn_like(output)
        loss: torch.Tensor = torch.nn.functional.mse_loss(output, target)

        loss.backward()

        time.sleep(0.1)

        return output

    def train_step(self) -> None:
        """
        Executes a single distributed training step over a global batch.

        This involves creating synthetic micro-batches, running them through the
        pipeline scheduler, and finally synchronizing gradients and updating weights.
        """
        self.logger.info("Starting training step.")

        self.optimizer.zero_grad()

        inputs: List[torch.Tensor] = [
            torch.randn(32, self.hidden_size, requires_grad=True)
            for _ in range(self.num_microbatches)
        ]

        self.pipeline_scheduler.execute(self._mock_forward_backward, inputs)

        self.dp_synchronizer.synchronize_gradients(list(self.model.parameters()))

        self.optimizer.step()

        self.logger.info("Completed training step.")
