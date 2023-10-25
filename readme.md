# VGG Model Training and Evaluation

This repository provides tools for training and evaluating VGG models. It includes scripts to train VGG models on a specific dataset and then evaluate/test the model.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Installation](#installation)

## Overview

- Model: VGG (Supporting VGG11, VGG13, VGG16, VGG19)
- Task: Classification
- Data: Custom Image Dataset

## Setup

1. Install Python 3.x and necessary libraries.
2. Clone this repository.

\```bash
git clone [repository URL]
cd [repository name]
\```

3. Install required packages.

\```bash
pip install torch torchvision tqdm
\```

## Usage

1. Place your data in the appropriate directory. For this example, the `data_clean` directory is referenced.
2. Train the model.

\```bash
python main.py
\```

3. Test the model.

\```bash
python test.py
\```

## Installation
```
pip install -r requirements.txt
```