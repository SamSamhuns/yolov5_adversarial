# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Image for adversarial patch generation and testing (train_patch.py / test_patch.py)
# Build: docker build --build-arg UID=$(id -u) -f adv_patch_gen/Dockerfile -t yolov5_adversarial:latest .

# Start FROM the official PyTorch image https://hub.docker.com/r/pytorch/pytorch/tags
# Override with a tag you have verified for a reproducible build, e.g.
#   --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ARG PYTORCH_IMAGE=pytorch/pytorch:latest
FROM ${PYTORCH_IMAGE}

# Install linux packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && TZ=Etc/UTC apt install -y --no-install-recommends tzdata \
    && apt install -y --no-install-recommends gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg \
    && apt upgrade -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

# set username inside docker
ARG UNAME=user1
ARG UID=1000

# add user UNAME as a member of the sudoers group
RUN useradd -rm --home-dir "/home/$UNAME" --shell /bin/bash -g root -G sudo -u "$UID" "$UNAME"

# Fonts the ultralytics Annotator needs, into the user's config dir rather than root's
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /home/$UNAME/.config/Ultralytics/
RUN chown -R "$UNAME" "/home/$UNAME/.config"

# activate user
USER "$UNAME"
ENV PATH="/home/$UNAME/.local/bin:${PATH}"

# Create working directory
WORKDIR "/home/$UNAME/app"

# Install pip packages. Export/logging extras are deliberately not installed, patch
# generation only needs the base yolov5 requirements plus the adv_patch_gen ones.
COPY requirements.txt .
COPY adv_patch_gen/requirements.txt adv_patch_gen/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel \
    && pip install --no-cache-dir -r requirements.txt -r adv_patch_gen/requirements.txt

# Set environment variables
ENV OMP_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy contents
# (might have issues as not a .git directory)
COPY --chown=$UNAME . "/home/$UNAME/app"
