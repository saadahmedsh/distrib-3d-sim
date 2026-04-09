import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple, Any
from src.topology.mesh import VirtualDeviceMesh
from src.engine.logger import DistributedLogger


class _CopyToTensorModelParallelRegion(torch.autograd.Function):
    """Pass-through function, does not modify forward pass, broadcasts gradients in backward pass."""

    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """
        Forward pass for copy operation.

        Args:
            ctx: Context object.
            input_tensor: Input tensor.
            group: Process group.

        Returns:
            The input tensor unmodified.
        """
        ctx.group = group
        return input_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        """
        Backward pass: all-reduce the gradient across the TP group.

        Args:
            ctx: Context object.
            grad_output: Gradient from the next layer.

        Returns:
            Tuple containing the synchronized gradient and None.
        """
        if ctx.group.size() > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _GatherFromTensorModelParallelRegion(torch.autograd.Function):
    """Gathers the input tensor across the tensor parallel group in the forward pass, splits in backward pass."""

    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """
        Forward pass: all-gather the input tensor across the TP group.

        Args:
            ctx: Context object.
            input_tensor: Input tensor.
            group: Process group.

        Returns:
            The gathered tensor.
        """
        ctx.group = group

        if group.size() == 1:
            return input_tensor

        tensor_list = [torch.empty_like(input_tensor) for _ in range(group.size())]
        dist.all_gather(tensor_list, input_tensor, group=group)

        output = torch.cat(tensor_list, dim=-1).contiguous()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], None]:
        """
        Backward pass for gather operation.

        Args:
            ctx: Context object.
            grad_output: Gradient from the next layer.

        Returns:
            The split input gradient and None.
        """
        if ctx.group.size() == 1:
            return grad_output, None

        rank = dist.get_rank(group=ctx.group)
        world_size = dist.get_world_size(group=ctx.group)

        dim_size = grad_output.size(-1)
        chunk_size = dim_size // world_size

        grad_input = torch.split(grad_output, chunk_size, dim=-1)[rank].contiguous()
        return grad_input, None


def copy_to_tensor_model_parallel_region(input_tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Copies input tensor to the tensor parallel region."""
    return _CopyToTensorModelParallelRegion.apply(input_tensor, group)


def gather_from_tensor_model_parallel_region(input_tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gathers input tensor from the tensor parallel region."""
    return _GatherFromTensorModelParallelRegion.apply(input_tensor, group)


class TensorParallelLinear(nn.Module):
    """
    A Linear layer that implements Column Parallelism.
    The weight matrix is sliced along the output dimension.
    """

    def __init__(
        self,
        mesh: VirtualDeviceMesh,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        """
        Initializes the TensorParallelLinear layer.

        Args:
            mesh: The VirtualDeviceMesh containing topology information.
            in_features: Number of input features.
            out_features: Total number of output features across all partitions.
            bias: Whether to include a bias term.

        Raises:
            ValueError: If out_features is not divisible by the TP size.
        """
        super().__init__()
        self.mesh: VirtualDeviceMesh = mesh
        self.tp_group: dist.ProcessGroup = mesh.get_tp_group()
        self.tp_size: int = mesh.tp_size
        self.logger: DistributedLogger = DistributedLogger(rank=mesh.rank)

        if out_features % self.tp_size != 0:
            raise ValueError("out_features must be divisible by tensor parallel size.")

        self.in_features: int = in_features
        self.out_features_per_partition: int = out_features // self.tp_size

        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(self.out_features_per_partition, self.in_features)
        )
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros(self.out_features_per_partition)
            )
        else:
            self.register_parameter('bias', None)

        self.logger.debug(
            f"Initialized TensorParallelLinear: in_features={self.in_features}, "
            f"out_features_per_partition={self.out_features_per_partition}"
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the column parallel linear layer.

        Args:
            input_tensor: The input tensor.

        Returns:
            The output tensor after the linear transformation and all-gather if necessary.
        """
        input_parallel: torch.Tensor = copy_to_tensor_model_parallel_region(
            input_tensor, self.tp_group
        )

        output_parallel: torch.Tensor = nn.functional.linear(
            input_parallel, self.weight, self.bias
        )

        output: torch.Tensor = gather_from_tensor_model_parallel_region(
            output_parallel, self.tp_group
        )

        return output
