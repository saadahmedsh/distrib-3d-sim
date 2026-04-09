import os
import sys
import time
import logging
from typing import Optional, List, Callable, Any, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

# =============================================================================
# 1. THE OBSERVER: DistributedLogger
# =============================================================================
# In distributed systems, multiple processes print to the same console.
# This logger ensures every message is tagged with the process's ID (its "Rank"),
# so you know exactly which virtual GPU is doing what.

class DistributedLogger:
    def __init__(self, rank: Optional[int] = None, name: str = "distrib-3d-sim") -> None:
        self.logger: logging.Logger = logging.getLogger(name + str(rank))
        self.logger.setLevel(logging.DEBUG)
        self.rank: Optional[int] = rank

        # Prevent duplicate logs if initialized multiple times
        if not self.logger.handlers:
            handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            # Format: Timestamp | Log Level | [Rank X] Message
            formatter: logging.Formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _format_message(self, message: str) -> str:
        if self.rank is not None:
            return f"[Rank {self.rank}] {message}"
        return message

    def info(self, message: str) -> None:
        self.logger.info(self._format_message(message))

    def debug(self, message: str) -> None:
        self.logger.debug(self._format_message(message))

    def error(self, message: str) -> None:
        self.logger.error(self._format_message(message))


# =============================================================================
# 2. THE TOPOLOGY: VirtualDeviceMesh
# =============================================================================
# This class creates the "3D Grid" of GPUs. It takes a flat list of 8 processes
# and organizes them into overlapping groups for Data, Tensor, and Pipeline parallelism.

class VirtualDeviceMesh:
    def __init__(self, world_size: int, rank: int, dp_size: int, tp_size: int, pp_size: int) -> None:
        if dp_size * tp_size * pp_size != world_size:
            raise ValueError("DP * TP * PP must equal world_size.")

        self.world_size = world_size
        self.rank = rank
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.logger = DistributedLogger(rank=rank)

        # 1. Initialize the global communication network using standard TCP/IP ('gloo')
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)
        
        self.dp_group: Optional[dist.ProcessGroup] = None
        self.tp_group: Optional[dist.ProcessGroup] = None
        self.pp_group: Optional[dist.ProcessGroup] = None

        # 2. Calculate which sub-groups this specific Rank belongs to
        self._create_process_groups()

    def _create_process_groups(self) -> None:
        # TENSOR PARALLEL GROUPS (Contiguous neighbors)
        # E.g., (0,1), (2,3), (4,5), (6,7)
        num_tp_groups = self.world_size // self.tp_size
        for i in range(num_tp_groups):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks=ranks)
            if self.rank in ranks:
                self.tp_group = group

        # PIPELINE PARALLEL GROUPS (Connecting stages)
        # E.g., (0,2), (1,3), (4,6), (5,7)
        for i in range(self.dp_size):
            for j in range(self.tp_size):
                ranks = [(i * self.pp_size * self.tp_size) + (k * self.tp_size) + j for k in range(self.pp_size)]
                group = dist.new_group(ranks=ranks)
                if self.rank in ranks:
                    self.pp_group = group

        # DATA PARALLEL GROUPS (Connecting model replicas)
        # E.g., (0,4), (1,5), (2,6), (3,7)
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                ranks = [(k * self.pp_size * self.tp_size) + (i * self.tp_size) + j for k in range(self.dp_size)]
                group = dist.new_group(ranks=ranks)
                if self.rank in ranks:
                    self.dp_group = group

    def get_dp_group(self) -> dist.ProcessGroup: return self.dp_group
    def get_tp_group(self) -> dist.ProcessGroup: return self.tp_group
    def get_pp_group(self) -> dist.ProcessGroup: return self.pp_group

    def cleanup(self) -> None:
        dist.destroy_process_group()


# =============================================================================
# 3. DATA PARALLELISM (DP): DataParallelSynchronizer
# =============================================================================
# Handles averaging the gradients across different replicas of the model.

class DataParallelSynchronizer:
    def __init__(self, mesh: VirtualDeviceMesh) -> None:
        self.mesh = mesh
        self.dp_group = mesh.get_dp_group()
        self.logger = DistributedLogger(rank=mesh.rank)

    def synchronize_gradients(self, parameters: List[torch.Tensor]) -> None:
        if self.mesh.dp_size <= 1: return

        self.logger.debug("DP: Averaging gradients across replicas...")
        for param in parameters:
            if param.grad is not None:
                # 1. Sum all gradients in the group
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                # 2. Divide by the number of replicas to get the mean
                param.grad.data.div_(self.mesh.dp_size)


# =============================================================================
# 4. PIPELINE PARALLELISM (PP): PipelineScheduler
# =============================================================================
# Splits the batch into "micro-batches" and coordinates sending them between 
# Stage 0 and Stage 1 to keep all GPUs busy.

class PipelineScheduler:
    def __init__(self, mesh: VirtualDeviceMesh, num_microbatches: int) -> None:
        self.mesh = mesh
        self.pp_group = mesh.get_pp_group()
        self.pp_rank = dist.get_rank(self.pp_group) # 0 for Stage 0, 1 for Stage 1
        self.pp_size = mesh.pp_size
        self.num_microbatches = num_microbatches
        self.logger = DistributedLogger(rank=mesh.rank)

    def execute(self, forward_step_fn: Callable[..., Any], inputs: List[Any]) -> None:
        self.logger.debug(f"PP: Starting pipeline with {self.num_microbatches} micro-batches.")

        # This loop simulates the "fill and drain" schedule.
        # Stage 0 starts right away. Stage 1 has to wait 1 step for Stage 0 to finish.
        for step in range(self.num_microbatches + self.pp_size - 1):
            if self.pp_rank == 0:
                if step < self.num_microbatches:
                    self.logger.debug(f"PP [Stage 0]: Processing micro-batch {step}")
                    forward_step_fn(inputs[step])
                    
            elif self.pp_rank == 1:
                if step >= 1 and (step - 1) < self.num_microbatches:
                    mb_idx = step - 1
                    self.logger.debug(f"PP [Stage 1]: Processing micro-batch {mb_idx} (received from Stage 0)")
                    forward_step_fn(inputs[mb_idx])


# =============================================================================
# 5. TENSOR PARALLELISM (TP): TensorParallelLinear & Functions
# =============================================================================
# Slices a single Neural Network layer across multiple GPUs.

class _CopyToTensorModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group):
        ctx.group = group
        return input_tensor # Forward: Do nothing, just pass the data

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group.size() > 1:
            # Backward: Sum the gradients from the split pieces
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None

class _GatherFromTensorModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group):
        ctx.group = group
        if group.size() == 1: return input_tensor
        # Forward: Gather the sliced outputs from partner GPUs and combine them
        tensor_list = [torch.empty_like(input_tensor) for _ in range(group.size())]
        dist.all_gather(tensor_list, input_tensor, group=group)
        return torch.cat(tensor_list, dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.group.size() == 1: return grad_output, None
        # Backward: Split the incoming gradient back into slices
        rank = dist.get_rank(group=ctx.group)
        world_size = dist.get_world_size(group=ctx.group)
        chunk_size = grad_output.size(-1) // world_size
        return torch.split(grad_output, chunk_size, dim=-1)[rank].contiguous(), None

# Helper functions to trigger the custom autograd rules above
def copy_to_tp_region(input_tensor, group): return _CopyToTensorModelParallelRegion.apply(input_tensor, group)
def gather_from_tp_region(input_tensor, group): return _GatherFromTensorModelParallelRegion.apply(input_tensor, group)

class TensorParallelLinear(nn.Module):
    def __init__(self, mesh: VirtualDeviceMesh, in_features: int, out_features: int) -> None:
        super().__init__()
        self.tp_group = mesh.get_tp_group()
        self.logger = DistributedLogger(rank=mesh.rank)
        
        # SLICE THE WEIGHTS: Each GPU only holds 1 / tp_size of the total weights!
        self.out_features_per_partition = out_features // mesh.tp_size
        self.weight = nn.Parameter(torch.randn(self.out_features_per_partition, in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_parallel = copy_to_tp_region(input_tensor, self.tp_group)
        # Local math using only the sliced weights
        output_parallel = nn.functional.linear(input_parallel, self.weight, self.bias)
        # Combine results with TP partner
        return gather_from_tp_region(output_parallel, self.tp_group)


# =============================================================================
# 6. THE ORCHESTRATOR: SimulatedTrainer
# =============================================================================
# Ties DP, PP, and TP together into a single training step.

class SimulatedTrainer:
    def __init__(self, mesh: VirtualDeviceMesh, hidden_size: int = 1024, num_microbatches: int = 4) -> None:
        self.mesh = mesh
        self.hidden_size = hidden_size
        self.num_microbatches = num_microbatches
        self.logger = DistributedLogger(rank=mesh.rank)

        # 1. Initialize TP Model
        self.model = TensorParallelLinear(mesh, hidden_size, hidden_size)
        # 2. Initialize DP Synchronizer
        self.dp_synchronizer = DataParallelSynchronizer(mesh)
        # 3. Initialize PP Scheduler
        self.pipeline_scheduler = PipelineScheduler(mesh, num_microbatches)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def _mock_forward_backward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Simulate computation time
        time.sleep(0.1) 
        output = self.model(input_tensor)
        
        # Fake loss calculation to trigger backward pass
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        time.sleep(0.1)
        return output

    def train_step(self) -> None:
        self.logger.info(">>> Starting Global Training Step")
        self.optimizer.zero_grad()

        # Create dummy data for the micro-batches
        inputs = [torch.randn(32, self.hidden_size, requires_grad=True) for _ in range(self.num_microbatches)]

        # 1. Execute Pipeline (which implicitly executes Tensor Parallelism inside the model)
        self.pipeline_scheduler.execute(self._mock_forward_backward, inputs)

        # 2. Synchronize Data Parallel gradients across replicas
        self.dp_synchronizer.synchronize_gradients(list(self.model.parameters()))

        # 3. Update model weights
        self.optimizer.step()
        self.logger.info("<<< Completed Global Training Step\n")


# =============================================================================
# 7. MULTIPROCESSING: Worker & Main
# =============================================================================

def worker(rank: int, world_size: int, dp_size: int, tp_size: int, pp_size: int, num_steps: int) -> None:
    # This function is executed inside each of the 8 separated processes.
    try:
        # Build the mesh identity for this specific process
        mesh = VirtualDeviceMesh(world_size, rank, dp_size, tp_size, pp_size)
        trainer = SimulatedTrainer(mesh, hidden_size=256, num_microbatches=2)

        for step in range(num_steps):
            if rank == 0: print(f"\n========== ITERATION {step + 1} ==========\n")
            # Ensure all processes start the step at the same time for clean logs
            time.sleep(0.2) 
            trainer.train_step()

    except Exception as e:
        print(f"Error on Rank {rank}: {e}")
    finally:
        mesh.cleanup()

def main() -> None:
    # --- 3D Configuration ---
    dp_size = 2 # 2 Model Replicas
    tp_size = 2 # Layers split into 2
    pp_size = 2 # Model split into 2 stages
    world_size = dp_size * tp_size * pp_size # 8 Total Processes
    num_steps = 2

    print(f"Initializing 3D Parallel Simulation...")
    print(f"Total Virtual GPUs: {world_size}")
    print(f"Topology: DP={dp_size}, TP={tp_size}, PP={pp_size}\n")

    # Force PyTorch to use 1 thread per process so our CPU doesn't melt
    os.environ["OMP_NUM_THREADS"] = "1"

    # Spawn 8 identical processes. They will differentiate themselves using their 'rank'.
    mp.spawn(
        worker,
        args=(world_size, dp_size, tp_size, pp_size, num_steps),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()