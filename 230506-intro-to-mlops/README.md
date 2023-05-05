# Intro to MLOps

## Getting Started

First, go to code seoul github repository [link](https://github.com/CodeSeoul/machine-learning) 
and clone the repository. You can find the target HTTPS / SSH link by clicking `code` to get the link.

```shell
git clone <link>
```

Run the following script to setup and install virtual env on your device.

```shell
cd machine-learning
python setup_venv.py -p 230506-intro-to-mlops
```

Afterwards, change directory into the appropriate folder.

```shell
cd machine-learning/230506-intro-to-mlops
```

### Acvitvate Virtual Environment

To prevent polluting the global package on your local device, 
I recommend using a virtual environment. The `setup_venv.py` should have 
installed all dependencies to your virtual environment. 
All we need to do is activate it.

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
