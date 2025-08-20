FROM python:3.10-bookworm

RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

# Install system dependencies (optional, for scientific computing)
#Usage:docker build -t physionet-challenge .

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir wfdb h5py pandas scikit-learn numpy

# If you have a requirements.txt, uncomment the following line:
# RUN pip install -r requirements.txt