FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Install development tools and dnf plugins
RUN dnf update -y && \
    dnf install -y \
    git \
    wget \
    tar \
    python3 \
    python3-pip \
    python3-devel \
    vim \
    sudo \
    gcc \
    gcc-c++ \
    dnf-plugins-core \
    # OpenCV dependencies
    mesa-libGL \
    libgomp \
    # Image processing libraries
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    # Scientific computing
    openblas-devel \
    lapack-devel \
    && dnf clean all

# Install Node.js 20.x (LTS)
RUN curl -fsSL https://rpm.nodesource.com/setup_20.x | bash - && \
    dnf install -y nodejs && \
    dnf clean all

# Create non-root user
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install CUDA toolkit
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y cuda-toolkit-12-6 && \
    dnf clean all

# Set CUDA environment variables
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

USER vscode
WORKDIR /workspace

# Install PyTorch with CUDA support
RUN pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124