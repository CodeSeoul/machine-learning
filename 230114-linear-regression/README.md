# Simple Linear Regression

## Dataset

We are using the [Salary_data](https://www.kaggle.com/datasets/vihansp/salary-data) which is available on Kaggle. 
It is a very small toy dataset and thus is ideal for learning linear regression.

## Getting Started 

First, after checking out the project from GitHub, navigate to the right folder

```shell
cd machine-learning/230114-linear-regression
```

### Install and activate Virtual Environment

To prevent polluting the global package on your local device, 
I recommend using a virtual environment. To install venv, type in 
the following:

```shell

# For unix/macos
python -m pip install --user virtualenv

# For windows
py -m pip install --user virtualenv
```

Afterwards, we need to create a virtual environment. 

```shell
# unix / macos
python -m venv linear_regression

# windows
py -m venv linear_regression
```

After creating a new virtual environment, we need to activate it :)

```shell
# unix / macos
source linear_regression/bin/activate

# windows
.\linear_regression\Scripts\activate
```

We can confirm that we are in the virtual environment by typing in 
`which python` (mac /unix) or `where python` for windows. It should be the 
`linear_regression` venv directory.

To exit the virtual environment, type `deactivate`. 
Shutting down the terminal window also works.

### Install Dependencies

After installing the venv, it is recommended to update pip with the following command

```shell
python -m pip install --upgrade pip
```

Afterwards, install the dependencies using the following command

```shell
python -m pip install -r requirements.txt
```

And you are now good to go!

## Implementing linear regression

Afterwards, open up `update_me.py` and work on implementing linear regression

