# Hidden Subdomain Adaptation with Variable Number of Subdomains (HSAV)


## Content

* ``./data`` contains the preprocessed datasets of Kaggle fraud detection tasks.
* ``./env`` provides a Dockerfile to build the executing environment with all packages.
* ``./logs`` saves the execution results.
* ``./model`` contains the pre-trained models.
* ``./notebooks`` gives jupyter notebooks of our experiments.
* ``./preprocessing`` preporcesses the raw data to get correct representation and prepares pre-trained source models.
* ``./src`` contains the source code of adaptation methods, utils, etc.

## Environment

We provide a docker file to reproduce the environment that we used.

To use our docker image, go to the folder ``./env`` and run
```
docker build -t hsav_env .
```
Once the docker image is built, run
```
docker run -it --rm --name hsav_exp --runtime=nvidia -v {project_path}:/opt/notebook -p {local_port}:11112 hsav_env
```
to run our exps in an interactive mode on your web browser. 

Then you can have access to our notebooks at `localhost:{local_port}`.

**The GPU support is required.**


## Run Exps

Notebooks of all exps are provided in ``./notebooks``. You can check the source codes in the folder ``./src``.

## Evaluation

We provide a notebook ``./notebooks/exp_analyse_kaggle.ipynb`` to visualize experimental results.

The folder ``./logs`` contains all experimental results.

## Data Preprocessing (Optional)

We provide the preprocessed Kaggle datasets in the folder ``./data``.

If you want to preprocess the dataset yourself, you need to go to the folder ``./preprocessing`` and run 
```
python kaggle_data_preprocessing.py
```
in order to get the correct represenations of the Kaggle Fraud Detection datasets.

Make sure to put downloaded raw files in the folder ``./data``.

The files we need are ``train_transaction.csv`` and ``train_identity.csv``. They can be found here:
> https://www.kaggle.com/c/ieee-fraud-detection/data


## Model Pre-training (Optional)
We provide the pre-trained source models in ``./model``.
If you want to run these steps by yourself, go to ``./preprocessing`` then run
```
python kaggle_uni_model.py
``` 
