

# Fake News Detection Project

## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Model Training](#model-training)
6. [Testing](#testing)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Description

This project focuses on the detection of fake news using machine learning techniques. It utilizes a dataset of news articles labeled as fake or true and builds models to classify new articles as either fake or not fake based on their content.

## Installation

To use this project, follow these steps to set up the required dependencies:

```bash
!pip install kaggle
!pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/competitions/fake-news")
od.download("https://www.kaggle.com/datasets/jainpooja/fake-news-detection")
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
```

## Usage

To use the fake news detection models, follow these steps:

1. Install the required dependencies as mentioned in the [Installation](#installation) section.
2. Load the dataset and preprocess it using the provided code.
3. Train the models (Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest) on the preprocessed data.
4. Evaluate the models using the test dataset.
5. Use the `manual_testing` function to test individual news articles manually.

## Dataset

The project utilizes two datasets for training and testing:

1. Fake News Competition Dataset: [https://www.kaggle.com/competitions/fake-news](https://www.kaggle.com/competitions/fake-news)
2. Fake News Detection Dataset: [https://www.kaggle.com/datasets/jainpooja/fake-news-detection](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)

Both datasets contain news articles labeled as fake or true.

## Model Training

The models used in this project are:

1. Logistic Regression
2. Decision Tree Classifier
3. Gradient Boosting Classifier
4. Random Forest Classifier

The text data is transformed into TF-IDF vectors for model training.

## Testing

The trained models are evaluated using a test dataset. The performance metrics, such as accuracy and classification report, are displayed for each model.

## Results

The results of the model evaluations are presented in the project. Metrics and visualizations may be provided to demonstrate the performance of each model.

## Contributing

Contributions to this project are welcome! If you want to contribute, please follow the standard guidelines for pull requests.

## License

This project is licensed under the [MIT License](LICENSE).

## How to Use the "manual_testing" Function

The `manual_testing` function allows you to manually test news articles for fake news detection. To use this function, follow these steps:

1. Call the `manual_testing` function with the news article as input.
2. The function will preprocess the input text and pass it through the trained models.
3. The function will output the predicted label: "Fake News" or "Not A Fake News."

```python
news = str(input())
manual_testing(news)
```

Remember to replace `news` with the actual news article you want to test.

---
