# Crop disease prediction and pesticide/or natural remedies recommendation

## Overview

This repository contains code for a crop disease prediction and remedies recommendation system. The Crop Disease Prediction and Remediation System plays a crucial role in modern agriculture by providing farmers with proactive tools to manage and safeguard their crops effectively. By accurately predicting the likelihood of crop diseases, the system empowers farmers to take preemptive measures, thereby reducing the risk of yield loss and economic damage. Additionally, the recommendation of appropriate remedies, whether pesticide-based or natural, enables farmers to make informed decisions that balance effectiveness with environmental sustainability. Ultimately, this system not only helps optimize crop yields but also contributes to the sustainable and responsible management of agricultural resources, ensuring food security for communities worldwide.

## About the Dataset
We have used Plant Diseases Training Dataset from Kaggle. This dataset contains a collection of images of various plant leaves affected by different diseases. It is ccontain 95868 images and 38 disease type.
> Total Images: 95868

> Classes: 38 (e.g., 'Apple___Apple_scab', 'Apple___Black_rot', 'Grape___Black_rot', etc.) 


## Repository Structure

* model: Contains saved files of the model.
* src: Includes scripts for different purposes:
* preprocessing.py: Defines hyperparameters used in the model.
* Load_and_EDA.py: Includes scripts for loading the data.
* train.py: Script for training the models.
* predict.py: Script for outcome of the models.
* notebook: Experimental notebooks used for analysis and development.
* feature_engineering.py: Script for the feature engineering.

## Getting Started

Clone this repository.
Set up a Python environment and install the necessary dependencies listed in requirements.txt.
Utilize the provided scripts in the src directory for model training, data preprocessing, etc.

## Set up your Environment

### **`macOS`** type the following commands : 

- For installing the virtual environment you can either use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```
Or ....
- Install the virtual environment and the required packages by following commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    
### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


   
## Usage

In order to train the model and store test data in the data folder and the model in models run:

**`Note`**: Make sure your environment is activated.

```bash
python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```