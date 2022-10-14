# SANDI: Self-supervised learning for antigen detection on multiplex images

The scripts for implementing and validating the SANDI pipeline.

## Installation

- Clone this repository

  ```bash
  git clone https://github.com/yuerua/SANDI.git
  ```

- Create an environment and install the required libraries

  ```bash
  conda env create -f env_gpu.yml
  conda activate SANDI_gpu
  ```

## Implementation

- Training and validating with the automated reference set expansion scheme

  ```bash
  python main.py
  ```

- Performance comparison across various annotation budgets

  ```bash
  python model_comparisons/train_generator.py
  ```

  



