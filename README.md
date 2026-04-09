# distrib-3d-sim

A Python-based distributed training simulator enabling 3D Parallelism (Data, Tensor, and Pipeline) using `torch.distributed` with the `gloo` (CPU) backend.

## Technical Architecture

The simulator mimics real-world distributed cluster execution using standard CPU processes as stand-ins for individual "Nodes" and "GPUs". The core operations are categorized as follows:

- **Virtual Topology (`src/topology/mesh.py`)**
  Establishes a 3D coordinate system to map local CPU processes to discrete Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel (PP) communication groups.

- **Data Parallelism (`src/parallelism/data.py`)**
  Synchronizes gradients across DP groups via manual `all_reduce` operations at the conclusion of backward passes.

- **Tensor Parallelism (`src/parallelism/tensor.py`)**
  Distributes a `torch.nn.Linear` layer's computation across processes. The model's weights are sliced along an axis corresponding to the TP dimension, scaling memory usage and parallelizing linear algebra primitives.

- **Pipeline Parallelism (`src/parallelism/pipeline.py`)**
  Coordinates execution among PP groups utilizing a micro-batching scheduler. Distributes intermediate activations and backward gradients chronologically to minimize processing bubble overheads.

- **Engine (`src/engine/`)**
  Provides a `SimulatedTrainer` to synthesize a full model forward-backward pass with computationally delayed timings (`time.sleep`). Features an advanced `DistributedLogger` tracking tensors chronologically without visual bloat.

## Installation

This project utilizes `uv` for dependency management. To set up the environment and configure dependencies:

```bash
uv sync
```

Alternatively, `uv run` handles bootstrapping automatically during execution.

## Usage

To execute the simulator using the default configuration (World Size = 8 processes, partitioned into 2 DP, 2 TP, 2 PP):

```bash
uv run python main.py
```

The logger outputs structured diagnostics detailing the flow of activations and gradients, providing visibility into standard 3D parallelism scheduling.

## Testing

To run the verification suite and ensure parallel consistency:

```bash
PYTHONPATH=. uv run pytest
```
