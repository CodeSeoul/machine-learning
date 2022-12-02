# Digit Classification with MNIST

## Dataset

Since individuals should be able to train the model on their local device, 
we use the MNIST dataset: a collection of gray-scale handwritten digits from 0-9.

## Getting Started (Docker)

Build the docker image locally

```shell
cd machine-learning/221202-classification
docker build -t classification-tutorial -f classification.Dockerfile .
```

Afterwards, start up the docker container:

```shell
# start up container
docker run -p 8888:8888 --name classification-tutorial -d classification-tutorial
# Get the logs to find out the url
docker logs classification-tutorial --tail 100
```

Then, we should be able to access the jupyter notebook after seeing the following prompt in the terminal

```shell
To access the notebook, open this file in a browser:
file:///root/.local/share/jupyter/runtime/nbserver-9-open.html
Or copy and paste one of these URLs:
http://5fd51f27b488:8888/?token=42c0b4c865d69c064cf1477256e3baf05d9a50c1c89e92cb
or http://127.0.0.1:8888/?token=42c0b4c865d69c064cf1477256e3baf05d9a50c1c89e92cb
```

Copy and paste: `http://127.0.0.1:8888/?token=42c0b4c865d69c064cf1477256e3baf05d9a50c1c89e92cb` to your browser and
you are good to go. 

Note: The token value will differ each time, so make sure to copy and paste the value from your terminal and not the `README.md`
