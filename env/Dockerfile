FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

COPY settings/ /opt/settings/

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
    vim \
    curl \
    bzip2 \
    dos2unix \
    patch \
    build-essential

RUN pip install cmake
RUN pip install Cython
RUN pip install --ignore-installed -r /opt/settings/requirements.txt
RUN echo "root:root" | chpasswd

RUN jupyter notebook --generate-config && \
    cp -r /opt/settings/jupyter_notebook_config.py ~/.jupyter/

CMD ["/bin/bash", "-c", "jupyter notebook --allow-root"]
