# SFU CMPT 340 Project Template -- Replace with project title
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Add a 1-2 line summary of your project here.

## Important Links

| [Timesheet](https://google.com) | [Slack channel](https://google.com) | [Project report](https://google.com) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

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
│   └── test/ # test scripts
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
python -m venv env
env\Scripts\activate
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

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 
