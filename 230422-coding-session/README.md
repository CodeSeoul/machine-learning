# Live Coding Session - Building a Classifier

## Getting Started 

First, go to code seoul github repository [link](https://github.com/CodeSeoul/machine-learning) 
and clone the repository. You can find the target HTTPS / SSH link by clicking `code` to get the link.

```shell
git clone <link>
```

Afterwards, change directory into the appropriate folder.

```shell
cd machine-learning/230422-coding-session
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
python -m venv venv

# windows
py -m venv venv
```

After creating a new virtual environment, we need to activate it :)

```shell
# unix / macos
source venv/bin/activate

# windows
.\venv\Scripts\activate
```

We can confirm that we are in the virtual environment by typing in 
`which python` (mac /unix) or `where python` for windows. It should be the 
`venv` venv directory.

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
