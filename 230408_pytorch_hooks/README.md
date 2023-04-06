# PyTorch hooks

## Getting Started 

First, go to code seoul github repository [link](https://github.com/CodeSeoul/machine-learning) 
and clone the repository. You can find the target HTTPS / SSH link by clicking `code` to get the link.

```shell
git clone <link>
```

Afterwards, change directory into the appropriate folder.

```shell
cd machine-learning/230408_pytorch_hooks
```

### Docker setup (optional)

If you don't like using virtual environments and prefer to work inside of docker, run the following commands. 

1. Build the docker image

```shell
docker build -t pytorch_hooks .
```

This might take a while if your internet connection / PC is slow, since we are also installing `PyTorch` (a very large package).

2. Run docker image

```shell
docker run --name pytorch_hooks -d -p 8888:8888 pytorch_hooks
```

What the command above does is it creates a new container named `pytorch_hooks` (`--name pytorch_hooks`) from the image `pytorch_hooks` in a detached state (`-d`) and maps the hosts port `8888` to the docker container's local `8888` port.

3. Copy and paste jupyter notebook uri

Type in `docker logs pytorch_hooks` and you will see something like the following: 

```shell
I 08:10:42.471 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[W 08:10:42.832 NotebookApp] Loading JupyterLab as a classic notebook (v6) extension.
[C 08:10:42.832 NotebookApp] You must use Jupyter Server v1 to load JupyterLab as notebook extension. You have v2.5.0 installed.
    You can fix this by executing:
        pip install -U "jupyter-server<2.0.0"
[I 08:10:42.834 NotebookApp] Serving notebooks from local directory: /app
[I 08:10:42.834 NotebookApp] Jupyter Notebook 6.5.3 is running at:
[I 08:10:42.834 NotebookApp] http://ugfusgfuad:8888/?token=<token>
[I 08:10:42.834 NotebookApp]  or http://127.0.0.1:8888/?token=<token>
[I 08:10:42.834 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 08:10:42.838 NotebookApp] 
    
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-7-open.html
    Or copy and paste one of these URLs:
        http://ugfusgfuad:8888/?token=<token>
     or http://127.0.0.1:8888/?token=<token>
[I 08:10:53.882 NotebookApp] 302 GET / (172.17.0.1) 0.600000ms
[I 08:10:53.886 NotebookApp] 302 GET /tree? (172.17.0.1) 0.730000ms
```

Copy and paste the url: `http://127.0.0.1:8888/?token=<token>` and you should see a juypter notebook webpage, which means you are good to go!

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
python -m venv pytorch_hooks

# windows
py -m venv pytorch_hooks
```

After creating a new virtual environment, we need to activate it :)

```shell
# unix / macos
source pytorch_hooks/bin/activate

# windows
.\pytorch_hooks\Scripts\activate
```

We can confirm that we are in the virtual environment by typing in 
`which python` (mac /unix) or `where python` for windows. It should be the 
`pytorch_hooks` venv directory.

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

Afterwards, type in `juypter notebook` inside of the terminal and you are good to go!
