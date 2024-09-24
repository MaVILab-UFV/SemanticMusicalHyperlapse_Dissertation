FROM rust:bullseye AS builder

WORKDIR /workspace
COPY . /workspace

RUN cargo build --release

FROM ubuntu:24.04

SHELL ["/bin/bash", "-c", "-l"]

ARG DEBIAN_FRONTEND=noninteractive
RUN <<EOF
apt update
apt install --yes build-essential git wget libxml2

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda
rm Miniforge3-Linux-x86_64.sh

/opt/conda/bin/conda init --system --all
EOF

COPY environment.yml /environment.yml
RUN <<EOF
conda env create --file /environment.yml --name smh
echo "conda activate smh" | tee --append /root/.bashrc
EOF

WORKDIR /workspace
COPY . /workspace

COPY --from=builder /workspace/target/release/smh-rs /workspace/examples/smh-rs

ENTRYPOINT ["/bin/bash", "-c", "-l"]
