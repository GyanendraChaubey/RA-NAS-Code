# Data Directory

This project expects datasets to be available under this directory.

## CIFAR-10

By default, `scripts/run_experiment.py` uses `torchvision.datasets.CIFAR10` and stores data in:

`./data/cifar-10-batches-py` (managed by torchvision)

If your environment has no internet access, the pipeline automatically falls back to `torchvision.datasets.FakeData` so experiments can still run in `mock_mode`.

## Custom Datasets

To use a custom dataset:

1. Add a new loader path in `scripts/run_experiment.py`.
2. Update `configs/train.yaml -> dataset.name`.
3. Keep output shape and `num_classes` consistent with `src/models/cnn.py`.

