ARG PYTHON_VERSION="3.11"

FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip --no-cache-dir install --upgrade pip setuptools

# Copy pyproject.toml for dependencies
COPY pyproject.toml .
RUN mkdir -p "src/plant_care_ai"

# Install PyTorch CPU version
RUN pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install project dependencies (dev includes all deps)
RUN pip install --no-cache-dir ".[dev,api]" && rm -rf "src"

# Copy source code and install package
COPY src ./src
RUN pip install --no-cache-dir -e .

# Final image
FROM python:${PYTHON_VERSION}-slim AS final

# Create non-root user for security
ENV USER=plant_user
ENV UID=1001
ENV GID=1001
ENV GROUP=plant_group
RUN addgroup --gid $GID $GROUP && adduser --uid $UID --gid $GID --disabled-password $USER

ENV HOME=/home/$USER
ENV WORKSPACE=$HOME/work
WORKDIR $HOME

# Copy virtual environment from builder
COPY --from=builder --chown=$UID:$GID /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code
COPY --chown=$UID:$GID pyproject.toml .
COPY --chown=$UID:$GID src ./src

# Install package in editable mode
RUN pip install -e "."

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create directories for models and data
RUN mkdir -p /app/models && chown -R $UID:$GID /app

# Switch to non-root user
USER $UID:$GID

# Expose API port
EXPOSE 8001

# Environment variables
ENV DEVICE=cpu
ENV MODEL_CHECKPOINT_PATH=/app/models/best.pth
ENV CLASS_MAPPING_PATH=/app/models/class_id_to_name.json

# Run uvicorn server
CMD ["uvicorn", "plant_care_ai.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
