FROM python:3.8-buster

WORKDIR /app
COPY . .

# install vim in case we need to code inside of the container
# and python dependencies
# Alternatively, to code outside of docker, we can perform volume mounting
# but for now, this will suffice.
RUN apt update && apt install -y vim && pip install -r requirements.txt

# Start jupyter notebook
ENTRYPOINT jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=8888
