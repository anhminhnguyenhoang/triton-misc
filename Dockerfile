# Build with:
# docker build -t sage-bench -f Dockerfile .
#

# using amdsiloai/pytorch-xdit would be better, but there is unknown issue with the rocprof-compute installation
ARG BASE_IMAGE=amdsiloai/pytorch-xdit:v25.13.1
# Using vllm image because it has rocprof-compute in packages
# ARG BASE_IMAGE=rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103

FROM ${BASE_IMAGE} AS base


RUN apt-get update && apt-get install -y libdw-dev

# if cmake under 3.22, install cmake 3.22
RUN if [ "$(cmake --version | awk '{print $3}')" < "3.22" ]; then \
    /usr/local/bin/python -m pip install --user 'cmake==3.22.0'; \
    export PATH=${HOME}/.local/bin:${PATH}; \
fi

WORKDIR /opt/

RUN git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
RUN cd rocm-systems && \
    git sparse-checkout init --cone && \
    git sparse-checkout set projects/rocprofiler-sdk && \
    git checkout develop && \
    cmake -B rocprofiler-sdk-build -DCMAKE_INSTALL_PREFIX=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm projects/rocprofiler-sdk && \
    cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)

RUN apt install rocprofiler-compute
RUN update-alternatives --install /usr/bin/rocprof-compute rocprof-compute /opt/rocm/bin/rocprof-compute 0
RUN python3 -m pip install --ignore-installed blinker -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt