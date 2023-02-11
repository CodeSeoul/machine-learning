# Introduction to numpy

## Getting Started 

First, after checking out the project from GitHub, navigate to the right folder

```shell
cd machine-learning/230211_intro_to_numpy
```

### Install and activate Virtual Environmentk

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
python -m venv 230211_intro_to_numpy

# windows
py -m venv 230211_intro_to_numpy
```

After creating a new virtual environment, we need to activate it :)

```shell
# unix / macos
source 230211_intro_to_numpy/bin/activate

# windows
.\intro_to_numpy\Scripts\activate
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
Open up `exercises.py` and update the source code by reading the instructions. 
Your goal is to pass all the test cases.

