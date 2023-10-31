# VGG Model Training and Evaluation

This repository provides tools for training and evaluating VGG models. It includes scripts to train VGG models on a specific dataset and then evaluate/test the model.

## Table of Contents

1. [Setup](#setup)
2. [Usage](#usage)

## Setup 

Note. This is only for GPU
1. Create Python 3.10 Environment
```bash
conda create -n [environment name] python=3.10 -y
conda activate [environment name]
```
2. Clone this repository.

```bash
git clone [repository URL]
cd [repository name]
```

3. Install required packages.

```bash
pip install -r requirements.txt
```

## Usage

1. Place your data in the appropriate directory.
2. Train and test the model.

```bash
python main.py
```

3. (Optional) Only test the model.

```bash
cd src
python test.py
```
