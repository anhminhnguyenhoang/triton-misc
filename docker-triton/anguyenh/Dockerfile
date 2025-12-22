ARG MUID=1000
ARG MGID=1000
ARG USERNAME=developer

FROM rocm/aigmodels-private:mi350-dsr1-triton-moe-triton-gemms-250608-tuned-gemm
# FROM rocm/pytorch:rocm6.4.2_ubuntu22.04_py3.10_pytorch_release_2.6.0
# FROM compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-npi-mi350:972_ubuntu22.04_py3.10_pytorch_release-2.6_gfx950_cf65c6f

ENV DEBIAN_FRONTEND=noninteractive

# Remove jenkins which also uses gid=1000.
RUN userdel jenkins

RUN apt-get update
RUN apt-get install -y netcat-openbsd

# Here is all personal preference, with a few notes on software you may find useful
RUN apt-get update && \
apt-get install -y vim zsh zsh-doc tmux git less man man-db mlocate graphviz cmake-curses-gui ccache bash-completion && \
apt-get install -y silversearcher-ag keychain htop rcm && \
apt-get install -y libbz2-dev liblzma-dev libssl-dev libreadline-dev

# Build ccls if you want to
# RUN cd /opt && git clone --depth=1 --recursive https://github.com/MaskRay/ccls && \
# mkdir -p /opt/ccls/build && cd /opt/ccls/build && \
# cmake .. -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_PREFIX_PATH=/usr/lib/llvm-${LLVM_VER} \
#     -DLLVM_INCLUDE_DIR=/usr/lib/llvm-${LLVM_VER}/include && \
# make -j8 && \
# make install && \
# cd /opt && rm -rf /opt/ccls

# Install Neovim if you want to
# RUN apt-get install -yqq ninja-build gettext cmake unzip curl build-essential && \
# wget --progress=bar:force:noscroll https://github.com/neovim/neovim/archive/refs/tags/v0.9.5.tar.gz && \
# tar -xvf ./v0.9.5.tar.gz && cd neovim-0.9.5 \
# make CMAKE_BUILD_TYPE=RelWithDebInfo && make install && \
# cd .. && rm v0.9.5.tar.gz && rm -rf neovim-0.9.5

# Arcanist is needed for upstream, rocm-debug-agent is useful when debugging
RUN apt-get -y install arcanist rocm-debug-agent && updatedb

# Remove pre-build triton version
RUN pip uninstall -y triton

ARG MUID
ARG MGID
ARG USERNAME

RUN addgroup --gid ${MGID} ${USERNAME}
RUN useradd -d /home/${USERNAME} -g ${MGID} --no-create-home -u ${MUID} --shell /usr/bin/bash ${USERNAME}
RUN adduser ${USERNAME} sudo
RUN adduser ${USERNAME} video
RUN adduser ${USERNAME} render
# The /var is only needed on lockhart
RUN mkdir /home/${USERNAME} /var/${USERNAME}
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME} /var/${USERNAME}
RUN chown -R ${USERNAME}:${USERNAME} /opt/conda

# Make sudo work without a password
RUN sed -i~ -e 's/%sudo\tALL=(ALL:ALL) ALL/%sudo\tALL=(ALL:ALL) NOPASSWD:ALL/g' /etc/sudoers

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME=/home/${USERNAME}

# Keep container alive
CMD tmux new-session -d && tail -f /dev/null
