# ME5920 Final Project: Robust Autonomous Driving System

**Deadline**: 7 May

## Project Title: 
**Robust Autonomous Driving System Using Deep Learning and Generative AI**

---

## Overview

This project implements a robust autonomous driving system using deep learning and generative AI techniques. The system consists of three main components:

1. **Behavior Cloning**: A CNN model that learns to drive by mimicking human driving behavior.
2. **Data Augmentation**: Generative techniques to create challenging driving conditions (night, rain, fog).
3. **Evaluation Framework**: Comprehensive evaluation of model performance in various conditions.

## Package Structure

```
autodrive/
├── __init__.py              # Package initialization
├── data/                    # Data handling module
│   ├── __init__.py
│   └── dataset.py           # Dataset classes and loaders
├── models/                  # Model architectures
│   ├── __init__.py
│   └── cnn.py               # CNN models (basic CNN and ResNet)
├── augmentation/            # Data augmentation techniques
│   ├── __init__.py
│   └── transforms.py        # Regular and environmental transforms
├── evaluation/              # Evaluation utilities
│   ├── __init__.py
│   └── metrics.py           # Metrics and visualization
└── utils/                   # Utility functions
    ├── __init__.py
    ├── logger.py            # Logging utilities
    └── trainer.py           # Model training utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/ME592-Final-Project.git
cd ME592-Final-Project
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .  # Install package in development mode
```

## Dataset

We use the Udacity Self-Driving Car dataset, which contains images from a car's front-facing camera along with steering, throttle, and brake values. The dataset is automatically downloaded when you run the training script.

### Dataset Options

By default, the training script uses the Udacity CH2_002 dataset. If that fails to download, you can use the `--use_fallback_datasets` flag to try alternative Udacity datasets:

```bash
python train.py --use_fallback_datasets
```

You can also specify your own dataset URL:

```bash
python train.py --dataset_url https://your-dataset-url.com/dataset.zip
```

The dataset should be in a zip file and should contain:
- An `IMG` directory with images
- A `driving_log.csv` file with columns: center, left, right, steering, throttle, brake, speed

If your dataset doesn't match this structure exactly, the script will attempt to adapt it.

## Training

To train a model with default settings (ResNet18 with basic augmentation):

```bash
python train.py
```

With custom settings:

```bash
python train.py --model resnet --pretrained --augment --env_augment --env_type rain --epochs 30 --batch_size 64
```

Available command-line arguments:

- `--data_dir`: Directory to store dataset (default: "data")
- `--dataset_url`: URL to download the dataset from
- `--val_split`: Validation split ratio (default: 0.2)
- `--model`: Model architecture ("basic_cnn" or "resnet", default: "resnet")
- `--pretrained`: Use pretrained weights for ResNet (flag)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--augment`: Use basic data augmentation (flag)
- `--env_augment`: Use environmental augmentation (flag)
- `--env_type`: Type of environmental augmentation ("rain", "fog", "night", default: "rain")
- `--severity`: Severity of environmental augmentation (0.0 to 1.0, default: 0.5)
- `--log_dir`: Directory to store logs (default: "logs")
- `--save_dir`: Directory to save checkpoints (default: "checkpoints")
- `--tensorboard`: Use TensorBoard for logging (flag)
- `--device`: Device to use for training (default: "cuda" if available, else "cpu")
- `--resume`: Path to a checkpoint to resume training from (default: None)

## Evaluation

To evaluate a trained model on different environmental conditions:

```bash
python evaluate.py --model resnet --checkpoint checkpoints/best_model.pth --img_dir data/extracted/IMG
```

Available command-line arguments:

- `--data_dir`: Directory containing the dataset (default: "data")
- `--csv_file`: CSV file with validation data (default: "data/val.csv")
- `--img_dir`: Directory containing images (required)
- `--model`: Model architecture ("basic_cnn" or "resnet", default: "resnet")
- `--checkpoint`: Path to model checkpoint (required)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--environments`: Environments to evaluate on (default: ["clean", "rain", "fog", "night"])
- `--severity`: Severity of environmental effects (default: 0.5)
- `--results_dir`: Directory to save results (default: "results")
- `--device`: Device to use for evaluation (default: "cuda" if available, else "cpu")

## Demo

To visualize model predictions on individual images:

```bash
python demo.py --model resnet --checkpoint checkpoints/best_model.pth --input path/to/image.jpg
```

To visualize predictions with environmental effects:

```bash
python demo.py --model resnet --checkpoint checkpoints/best_model.pth --input path/to/image.jpg --env_type rain --severity 0.7
```

Available command-line arguments:

- `--model`: Model architecture ("basic_cnn" or "resnet", default: "resnet")
- `--checkpoint`: Path to model checkpoint (required)
- `--input`: Path to input image or directory of images (required)
- `--env_type`: Type of environmental augmentation to apply (None, "rain", "fog", "night")
- `--severity`: Severity of environmental augmentation (0.0 to 1.0, default: 0.5)
- `--output_dir`: Directory to save output visualizations (default: "demo_output")
- `--show`: Show visualizations in addition to saving them (flag)
- `--device`: Device to use for evaluation (default: "cuda" if available, else "cpu")

### Person 1:
- Set up and use simulator (CARLA or Udacity simulator)
- Record clean daytime driving data
- Build baseline CNN model
- Train and validate the baseline model on clean data

### Person 2: 
- Implement image augmentation to simulate night, rain, fog conditions
- Augment the clean dataset to create challenging conditions dataset
- Retrain the behavior cloning model on augmented + clean data
- Track model improvement after augmentation

### Person 3:
- Design and perform evaluations
- Write the final report in research paper format
- Prepare presentation slides
- Maintain GitHub repo

## License

This project is licensed under the MIT License.
