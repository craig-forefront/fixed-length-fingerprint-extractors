FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install all packages in one step
RUN dnf install -y \
    dnf-plugins-core \
    git \
    wget \
    tar \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    vim \
    sudo \
    gcc \
    gcc-c++ \
    # OpenCV dependencies
    mesa-libGL \
    libgomp \
    # Install Node.js
    nodejs \
    npm \
    # Image processing libraries
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    # Scientific computing
    openblas-devel \
    lapack-devel \
    && dnf clean all

# Create non-root user
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install CUDA toolkit
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y cuda-toolkit-12-8 && \
    dnf clean all

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Switch to vscode user
USER vscode
WORKDIR /workspace

# Create Python virtual environment
RUN python3.11 -m venv /home/vscode/.venv

# Activate venv and install PyTorch 2.9+ with CUDA 12.8 support
RUN /home/vscode/.venv/bin/pip install --upgrade pip && \
    /home/vscode/.venv/bin/pip install "torch>=2.9.0" "torchvision>=0.20.0" torchaudio --index-url https://download.pytorch.org/whl/cu128

# Add venv to PATH
ENV PATH="/home/vscode/.venv/bin:${PATH}"