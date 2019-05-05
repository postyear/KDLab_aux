# PV056-AutoML-testing-framework
* *PV056 Machine learning and knowledge discovery*

## How to use PV056-AutoML-testing-framework
* First, follow the [Installation guide](#installation-guide) section
* Then follow the [Usage](#usage) section
 
## Installation guide 
### Prerequisites
- Python version >=3.6 (including python3-dev, python3-wheel)
- [Weka](https://www.cs.waikato.ac.nz/ml/weka/)

### Installation to python virtual env (recommended)
It's highly recommended to install this testing framework to python virtual environment.
- Simple python virtual environment guide: https://realpython.com/python-virtual-environments-a-primer/

Simply run commands below in the root folder of this repository.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -e .
```

### Installation without python virtual env (not recommended)
```
$ pip install .
```


### Downloading datasets
All data files are from [OpenML](https://www.openml.org).

Data files are compressed in `data/datasets/openML-datasets.zip` (you have to unzip it). Because this file is larger than 50mb we are using git lfs (large file storage). You can read the documentation [here](https://git-lfs.github.com).

#### TL;DR
Run commands below in the root folder of this repo.
```
$ sudo apt install git-lfs
$ git lfs install
$ git lfs pull
$ cd data && unzip openML-datasets.zip
```


## Usage
If you have chosen to install this tester in the virtual environment, you must activate it to proceed.
* *Pipeline 1:* split data -> run classifiers -> (optional) statistics
* *Pipeline 2 (outlier detection):* split data -> apply outlier detectors -> remove outliers -> run classifiers -> (optional) statistics


### Split data
```
(venv)$ pv056-split-data --help
usage: pv056-split-data [-h] --config-file CONFIG_FILE --datasets-file
                        DATASETS_FILE

Script splits datasets for cross-validation

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-file DATASETS_FILE, -d DATASETS_FILE
                        Filename of output datasets config
```
#### Example usage
```
(venv)$ pv056-split-data -c config_split_example.json -d datasets.csv
```

#### Example config file
* *data_path*
    * Directory with datasets in arff format
* *train_split_dir*
    * Directory where generated **train** datasets should be saved
* *test_split_dir*
    * Directory where generated **test** datasets should be saved

```json
{
    "data_path": "data/datasets/",
    "train_split_dir": "data/train_split/",
    "test_split_dir": "data/test_split/"
}
```

### Apply outlier detection methods
```
(venv)$ pv056-apply-od-methods --help
usage: pv056-apply-od-methods [-h] --config-file CONFIG_FILE

Apply outlier detection methods to training data

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
```
#### Example usage
```
(venv)$ pv056-apply-od-methods -c config_apply_od_example.json
```

#### Example config file
* *train_split_dir*
    * Directory with splitted **train** datasets
* *train_od_dir*
    * Directory where generated **train** datasets with outlier detection values should be saved
* *od_methods*
    * List with Outlier detection methods
    * Outlier detection method schema:
        * *name* - OD name
        * *parameters* - Dictionary "parameter_name": "value"

```json
{
    "train_split_dir": "data/train_split/",
    "train_od_dir": "data/train_od/",
    "od_methods": [
        {
            "name": "IsolationForest",
            "parameters": {
                "contamination": "auto",
                "behaviour": "new"
            }
        },
        {
            "name": "LOF",
            "parameters": {
                "contamination": "auto"
            }
        }
    ]
}
```

#### Outlier detector names and parameters:
| Name | Full name | Parameters |
|:----:|:----------:|:----------:|
| **LOF** | Local Outlier Factor | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) |
| **NearestNeighbors** | Nearest Neighbors | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) |
| **IsolationForest** | Isolation Forest | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
| **ClassLikelihood** | Class Likelihood | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
| **ClassLikelihoodDifference** | Class Likelihood Difference | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
| **F2** | Max individual feature efficiency | -- |
| **F3** | Maximum Individual Feature Efficiency | -- |
| **F4** | Collective Feature Efficiency | -- |
| **T1** | Fraction of maximum covering spheres | -- |
| **T2** | Ave number of points per dimension | -- |
| **MV** | Minority value | -- |
| **CB** | Class balance | -- |
| **IsolationForest** | Isolation Forest | [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) |
| **KDN** | K-Disagreeing Neighbors | n_neighbors |
| **DS** | Disjunct size | -- |
| **DCP** | Disjunct class percentage | min_impurity_split [docs](https://blog.nelsonliu.me/2016/08/05/gsoc-week-10-scikit-learn-pr-6954-adding-pre-pruning-to-decisiontrees/) |
| **TD** | Tree Depth with and without prunning | -- |
| **TDWithPrunning** | Tree Depth with prunning | min_impurity_split |


* New methods for outlier detection coming soon!


### Remove outliers
```
(venv)$ pv056-remove-outliers  --help
usage: pv056-remove-outliers [-h] --config-file CONFIG_FILE --datasets-file
                             DATASETS_FILE

Removes the percentage of the largest outliers.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        JSON configuration
  --datasets-file DATASETS_FILE, -d DATASETS_FILE
                        Filename of output datasets config
```
#### Example usage
```
(venv)$ pv056-remove-outliers  -c config_remove_outliers_example.json -d datasets.csv
```

#### Example config file
* *test_split_dir*
    * Directory with splitted **test** datasets
* *train_od_dir*
    * generated **train** datasets with outlier detection values
* *train_removed_dir*
    * Directory where train data with **removed** outliers should be saved
* *percentage*
    * How many of the largest outliers should be removed (0-100)
```json
{
    "test_split_dir": "data/test_split/",
    "train_od_dir": "data/train_od/",
    "train_removed_dir": "data/train_removed/",
    "percentage": 10
}
```



### Run weka classifiers
To run a weka classifier using this framework, first setup virtual environment, install required modules and download weka tool.
1) Activate your virtual Python environment with this project.
2) Generate `datasets.csv` file using `pv056-split-data` or `pv056-remove-outliers` (See [Split data](#split-data) and [Remove outliers](#remove-outliers) )
3) Create a `config_clf_example.json` file, with weka classifiers and their configuration (See [Config file for weka classifiers](#example-of-config-file-for-weka-classifiers))
5) Run `pv056-run-clf` script, see command below

```
(venv)$ pv056-run-clf --help
usage: pv056-run-clf [-h] -c CONFIG_CLF -d DATASETS_CSV

PV056-AutoML-testing-framework

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_CLF, --config-clf CONFIG_CLF
                        path to classifiers config file
  -d DATASETS_CSV, --datasets-csv DATASETS_CSV
                        Path to csv with data files
```

#### Example usage
```
(venv)$ pv056-run-clf -c config_clf_example.json -d datasets.csv
```

#### Example of config file for weka classifiers
* *output_folder*
    * path to output folder, where outputs from your classifiers will be saved
* *weka_jar_path*
    * path to a weka.jar file
* *n_jobs*
    * number of parallel workers
* *classifiers*
    * list of classifiers which you want to run
    * you can run an arbitrary number of classifiers, even same classifier with different configuration
    * list of [weka classifiers](http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html)
    * *class_name*
        * name of weka classifier class
    * *args*
        * optional value
        * list of arguments for specific classifier
        * you can find all arguments for specific classifier using weka command: ```$ java -cp weka.jar weka.classifiers.trees.J48 --help``` or in [weka documentation](http://weka.sourceforge.net/doc.dev/weka/classifiers/trees/J48.html)
        * you can find more information about Weka-CLI in the section below [How to work with Weka 3](#how-to-work-with-Weka-3), but I don't think you need that for using this tool.
    * *filters*
        * optional value
        * you can use any filter from Weka
        * you have to specify name of filter and arguments for it

```json
{
    "output_folder": "clf_outputs/",
    "weka_jar_path": "weka-3-8-3/weka.jar",
    "n_jobs": 5,
    "classifiers": [
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C",
                0.25,
                "-M",
                2
            ],
            "filters": [
                {
                    "name": "weka.filters.unsupervised.attribute.RemoveByName",
                    "args": [
                        "-E",
                        "^size$"
                    ]
                }
            ]
        },
        {
            "class_name": "weka.classifiers.trees.J48",
            "args": [
                "-C",
                0.35
            ]
        },
        {
            "class_name": "weka.classifiers.bayes.BayesNet"
        }
    ]
}
```

### Count accuracy
To count accuracy simply run `pv056-statistics` script. In the future, we will add Precision and Recall.
```
(venv)$ pv056-statistics --help
usage: pv056-statistics [-h] --results-dir RESULTS_DIR [--pattern PATTERN]

Script for counting basic statistic (Accuracy, )

optional arguments:
  -h, --help            show this help message and exit
  --results-dir RESULTS_DIR, -r RESULTS_DIR
                        Directory with results in .csv
  --pattern PATTERN, -p PATTERN
                        Regex for filename (Python regex)
```
#### Example
```
(venv)$ pv056-statistics -r clf_outputs/ -p "teaching.*"
teachingAssistant BayesNet 3e408e23621de037f4751689311eb00d.csv
         Accuracy: 0.9073
teachingAssistant J48 81498a187313e89f240c8ead4557906b.csv
         Accuracy: 0.5232
teachingAssistant J48 9f0cf2e85982a05ecf632ee428274ec3.csv
         Accuracy: 0.5166
```

## How to work with Weka 3
* Download Weka from https://www.cs.waikato.ac.nz/ml/weka/downloading.html
* Weka classifiers http://weka.sourceforge.net/doc.dev/weka/classifiers/Classifier.html
* Documentation to Weka in `The WEKA Workbench` & `Weka manual` documents at https://www.cs.waikato.ac.nz/ml/weka/documentation.html
```shell
# Example of running J48 classifier with diabetes dataset
java -cp weka.jar weka.classifiers.trees.J48 -t data/diabetes.arff

# General & Specific configuration options
java -cp weka.jar weka.classifiers.trees.J48 --help
```

## Developers guide

As developers, we have chosen to use [Black](https://github.com/ambv/black/) auto-formater and [Flake8](https://gitlab.com/pycqa/flake8) style checker. Both of these tools are pre-prepared for pre-commit. It's also recommended to use [mypy](https://github.com/python/mypy).


Since there is the typing module in the standard Python library, it would be a shame not to use it.  A wise old man once said: More typing, fewer bugs. [Typing module](https://docs.python.org/3/library/typing.html)


To prepare dev env run commands below.
```
$ python3 -m venv venv
$ source venv/bin/activate # venv/bin/activate.fish # for fish shell
(venv)$ pip install -r requirements.txt
(venv)$ pip install -r requirements-dev.txt
(venv)$ pre-commit install
(venv)$ pip install -e .
```

For generating `requirements.txt` we are using pip-compile from [pip-tools](https://github.com/jazzband/pip-tools).
For keeping your packages updated, use `pip-sync requirements.txt requirements-dev.txt`.
