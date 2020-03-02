FROM ubuntu:18.04 as base
RUN apt-get update && apt-get -y update
RUN apt-get install build-essential -y
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]

RUN mkdir src
WORKDIR src/
COPY . .
# Install Peddy
RUN INSTALL_PATH=~/anaconda \
    && wget https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh && bash Anaconda3-5.3.0-Linux-x86_64.sh -bfp /usr/local

RUN rm Anaconda3-5.3.0-Linux-x86_64.sh
RUN conda update -n base -c defaults conda
RUN conda create -n neurondemix python=3.6 trefide=1.2 -c jw3132 -c conda-forge

RUN activate neurondemix

# CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

RUN echo "source activate neurondemix" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN chmod +x ./start.sh && ./start.sh