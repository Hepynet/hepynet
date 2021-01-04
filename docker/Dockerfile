FROM python:3.8

# required packages
RUN apt-get update \
    && pip install --upgrade pip \
    && pip install numpy \
    && pip install matplotlib \
    && pip install scikit-learn \
    && pip install eli5 \
    && pip install pandas \
    && pip install seaborn \
    && pip install hyperopt

# install tensorflow & keras
RUN pip install tensorflow==2.2.0 \
    && pip install keras==2.3.1

# Install hepynet
RUN cd /opt \
    && git clone https://github.com/StarPrecursor/hepynet.git \
    && cd hepynet \
    && git checkout iss13-prepare-for-new-release \
    && pip install .

RUN echo "cd /work" >> /root/.bashrc

ENTRYPOINT /bin/bash