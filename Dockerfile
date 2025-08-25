# syntax=docker/dockerfile:1.4

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install required system packages and clean up
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN mkdir -p /home/appuser && chown appuser:appuser /home/appuser

ENV HOME=/home/appuser

# Install pip, setuptools, and wheel, and upgrade
RUN pip install --upgrade pip setuptools wheel

# Install Jupyter, matplotlib and ngsolve and cache pip packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install notebook jupyterlab webgui_jupyter_widgets matplotlib ngsolve

# Copy your project files into the container
COPY . /app
RUN chown -R appuser:appuser /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
USER appuser

# Add folder and subfolders to Python path
ENV PYTHONPATH /app

# Expose the port that the application listens on.
EXPOSE 8000

# Start Jupyter when the container runs
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root", "--notebook-dir=/app"]