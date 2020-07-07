FROM tensorflow/tensorflow:latest-gpu-py3

# Set up ROOT
RUN apt-get update \
&& apt-get install -y vim git dpkg-dev cmake g++ gcc binutils libx11-dev \
libxpm-dev libxft-dev libxext-dev python-dev wget gzip \
&& cd /opt \
&& wget https://root.cern/download/root_v6.20.04.Linux-ubuntu18-x86_64-gcc7.5.tar.gz \
&& gunzip root_v6.20.04.Linux-ubuntu18-x86_64-gcc7.5.tar.gz \
&& tar xvf root_v6.20.04.Linux-ubuntu18-x86_64-gcc7.5.tar \
&& cd root \
&& echo ". /opt/root/bin/thisroot.sh" >> /root/.bashrc

# Basic packages for training
RUN apt-get update \
&& pip install --upgrade pip \
&& pip install numpy \
&& pip install matplotlib \
&& pip install keras \
&& pip install scikit-learn \
&& pip install uproot

# Set up Jupyter
RUN pip install jupyterlab \
&& jupyter notebook --generate-config \
&& echo "c.NotebookApp.allow_root=True" >> /root/.jupyter/jupyter_notebook_config.py

# Other required packages
RUN pip install eli5 \
&& pip install ConfigParser \
&& pip install reportlab \
&& pip install pandas \
&& pip install seaborn \
&& pip install hyperopt

# Start up
RUN echo "pip install -e /work" >> /root/.bashrc \
&& echo "alias ghome='cd /work'" >> /root/.bashrc \
&& echo "alias lhome='cd /data'" >> /root/.bashrc \
&& echo "alias docker-jupyter='jupyter lab --notebook-dir=/work/run --ip 0.0.0.0 --no-browse'" >> /root/.bashrc \
&& echo "cd /work" >> /root/.bashrc

EXPOSE 8888
