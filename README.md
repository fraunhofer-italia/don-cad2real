# Dense Object Nets Training

Training and evaluation framework for **Dense Object Nets (DON)** dense visual correspondence pipelines.

Repository linked to the *Efficient Adaptation of Dense Object Descriptors with application to CAD-to-real Robotic Grasping* paper where the implemented DONs training, evaluation and fine-tuning pipelines are detailed. 

## Main features

- Dense descriptor networks training with different backbones and learning objectives.
- Optional sim-to-real fine-tuning of synthetic-only pretrained models based on weakly annotated real data.
- Various evaluation, interactive descriptor matching, and plotting utilities.

## Installation

The recommended environment is the Docker image defined by the repository.

Requirements:

- Docker
- Docker Compose
- NVIDIA drivers and NVIDIA Container Toolkit for GPU execution

Clone the repository and enter the Docker directory:

```bash
git clone https://github.com/fraunhofer-italia/don-cad2real.git
cd don-cad2real/docker
```

Build and run the Docker container:

```bash
docker compose build don_cad2real_docker
docker compose up -d don_cad2real_docker
```

The Compose configuration runs the container with the host UID/GID so that files created inside bind-mounted directories remain writable from the host. If the host UID or GID differs from the configured defaults, set the corresponding values in the Compose environment or `.env` file before starting the container.

GUI-based evaluation and visualization scripts require access to the host X11 display.

## Repository structure

The main parts of the repository are:

```text
.
├── dense_objects_nets/
│   ├── dataset/               # Dataset loading and augmentations
│   ├── evaluation/            # Shared evaluation infrastructure and metrics
│   ├── network/               # DON and task-specific network definitions
│   ├── pipelines/
│   │   └── pixels_matching/   # Dense correspondence training/evaluation
│   ├── training/              # Generic training loop
│   └── utils/                 # I/O, geometry, visualization, and helpers
├── config/
│   ├── dataset/               # Train/validation/test dataset definitions
│   ├── training/              # Pipeline-specific training parameters
│   └── training_runs/         # Complete training runs definitions
├── scripts/
│   ├── datasets_converters/   # Dataset conversion/manipulation tools
│   ├── evaluation/            # Standalone evaluation tools
│   ├── visualization/         # Metrics and descriptor visualization
│   └── misc/                  # Additional utilities
└── training_script.py         # Main training entry point
```

Runtime directories such as `data/`, `models/`, and `runs/` are mounted into the container by Docker Compose.

## Configuration

Training runs components and parameters are defined through YAML files under `config/`.

The main entry point is `config/training_runs/`. A training-run file selects and combines:

- a training pipeline;
- a network architecture;
- train, validation, and evaluation datasets;
- a pipeline-specific training configuration;
- optional pretrained checkpoints;
- optional parameter overrides or hyperparameter sweeps.

Use these files as references when defining new experiments:

```text
config/training_runs/_training_run_example.yml
config/training/_training_example.yml
config/dataset/_dataset_example.yml
```

Paths inside configuration files can use environment variables such as `${WS_DIR}`.

## Training

`training_script.py` is the main entry point for executing the experiments defined under `config/training_runs/`:

```bash
python training_script.py --config=<training-run-config>.yml
```

A training run assembles the selected dataset, network, trainer, and evaluator and writes its outputs under `runs/`.

Depending on the configuration, outputs include model checkpoints, copies of the resolved training and dataset configurations, evaluation configuration, quantitative metrics, and TensorBoard logs.

Example experiment definitions are available in `config/training_runs/`, including configurations for T-LESS, YCB-V, synthetic-to-real experiments, and small paper/demo setups.


## Evaluation

Standalone evaluation tools are available under `scripts/evaluation/`.

The main utilities are:

- `don_dataset_evaluator.py` — evaluate trained models quantitatively on one or more datasets;
- `don_predictions_evaluator.py` — inspect evaluation metrics and predictions interactively;
- `don_descriptors_matcher.py` — interactively inspect dense descriptor correspondences;
- `gpu_evaluator.py` — benchmark/evaluate GPU inference workloads.

Training runs save an `evaluation_config.yml`, which is used by the evaluation scripts to reconstruct the corresponding network, evaluator, and checkpoints.

Additional plotting tools are available under `scripts/visualization/` for comparing metrics, PCK curves, descriptor outputs, and dataset statistics.