FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3

# required packages
RUN apt-get update \
    && pip install --upgrade pip \
    && pip install numpy \
    && pip install matplotlib \
    && pip install keras==2.3.1 \
    && pip install scikit-learn \
    && pip install eli5 \
    && pip install pandas \
    && pip install seaborn \
    && pip install hyperopt

# Install hepynet
RUN git clone --recursive git@github.com:StarPrecursor/hepynet.git \
    && git checkout iss13-prepare-for-new-release \
    && pip install .
