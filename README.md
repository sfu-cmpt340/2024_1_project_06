# Automated Classification and Cell Counting of Lung Cancer from Histopathological Images Using Machine Learning (LPAI)

LPAI integrates a convolutional neural network model in PyTorch for automatic classification and quantification of lung cancer cells in histopathological images. This project implements preprocessing for enhanced image analysis, includes functionalities for model training, performance evaluation, and provides visualization of predictions alongside cell counts.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EToNF1db5yZNpQtMwcoy6jMBPKuCop_0wlRcKKu9BY7fyg?e=n970iV) | [Slack channel](https://sfucmpt340spring2024.slack.com/archives/C06EBMFFF3K) | [Project report](https://www.overleaf.com/project/65a57e2aa9883102c00b2515) |
|-----------|---------------|-------------------------|


## Video/demo/GIF
[![CMPT340 Lung Histopathology Classification](https://img.youtube.com/vi/lGQG9MRGoEw/0.jpg)](https://www.youtube.com/watch?v=lGQG9MRGoEw "CMPT340 Lung Histopathology Classification")

## Table of Contents
1. [Demo](#demo)

2. [Dataset Setup](#dataset-setup)

3. [Installation](#installation)

4. [Reproducing this project](#repro)

<a name="demo"></a>
## 1. Demo - Example output

After executing `src/train.py` in a virtual environment, here are some example outputs:

### Confusion Matrix on Max Subset Size of 5000 Images
![Confusion Matrix](https://i.imgur.com/LVdVacu.png)

### Classification Report for the Same Subset
The classification report below corresponds to the confusion matrix above, summarizing the model's performance on the same subset of 5000 images. It was generated using the scikit-learn library's `classification_report` function, reflecting the precision, recall, f1-score, and support for each class based on the model's predictions within the virtual environment terminal:

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| lung_aca  | 0.98      | 0.93   | 0.95     | 336     |
| lung_n    | 0.99      | 1.00   | 1.00     | 344     |
| lung_scc  | 0.94      | 0.97   | 0.96     | 320     |
| **accuracy**  |           |        | 0.97     | 1000    |
| **macro avg** | 0.97      | 0.97   | 0.97     | 1000    |
| **weighted avg** | 0.97      | 0.97   | 0.97     | 1000    |

Overall Accuracy: 0.97

### Cell Detection and Localization in Histopathological Images
![Cell Counting](https://i.imgur.com/5A6Omap.png)

### Predicted vs. Actual Classification Visualization

The visualization below showcases the model's predictions compared to the actual classifications for four randomly selected histopathological images from the loaded data. Each image is accompanied by a label indicating the true class and the class predicted by the model, along with the estimated cell count detected within the image. 

![Confusion Matrix](https://i.imgur.com/gcYj0lA.png)

<a name="dataset-setup"></a>
## 2. Dataset setup

Training data from Kaggle can be found [here](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images?resource=download). 

After downloading the datasets, unzip, then move lung_image_sets into your local repository as below.

```bash
2024_1_project_06/
│
├── data/ # upload training data downloaded from Kaggle
│   └── lung_image_sets/
│       ├── lung_aca/
│       ├── lung_n/
│       └── lung_scc/
│
├── src/
│   ├── __init__.py # makes src a Python package
│   ├── train.py # training script
│   └── test/ 
│       └── test.py
│
├── LICENSE
├── README.md
└── requirements.txt
```

<a name="installation"></a>

## 2. Installation

```bash
git clone https://github.com/sfu-cmpt340/2024_1_project_06.git
cd 2024_1_project_06
```
Create a virtual environment and activate it:

For Windows:
```bash
python -m venv env
env\Scripts\activate
```

For Unix systems (including Linux and macOS):
```bash
python -m venv env
source env/bin/activate
```

Install PyTorch by selecting the appropriate installation command from the [PyTorch official Get Started page](https://pytorch.org/get-started/locally/). The command you choose should correspond to your operating system, package manager (pip), Python version, and whether you need CUDA support. 

For example, if you're installing PyTorch with CUDA 11.8 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```
After PyTorch is installed, install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

To start training the main script, run `train.py`
```bash
python src/train.py
```

<a name="repro"></a>
## 3. Reproduction

To reproduce the results of the LPAI project after cloning and setting up your development environment:

1. Set up your dataset and follow the installation process.
3. Navigate to the `src` directory within your project repository.
4. To start training the model, run the `train.py` script:
```bash
python src/train.py
```

The script train.py is configured with predefined parameters for training. If you need to adjust settings such as the total_subset_size or the number of epochs, you will have to do so directly within the script.

For example, to change the total_subset_size, locate and modify the following line in train.py:
```bash
train_loader, test_loader, class_names = load_data(total_subset_size=400)  # Adjust this number as needed
```
After making the necessary changes, save the script and execute it as shown above. The training process will begin using the new subset size you specified.
