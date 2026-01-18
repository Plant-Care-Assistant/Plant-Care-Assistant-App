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

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install project dependencies (dev includes all deps)
RUN pip install --no-cache-dir ".[dev]" && rm -rf "src"

# Install FastAPI and uvicorn for API server
RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.0 \
    python-multipart==0.0.20

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

# Create directories for models and data
RUN mkdir -p /app/models && chown -R $UID:$GID /app

# Switch to non-root user
USER $UID:$GID

# Expose API port
EXPOSE 8001

# Environment variables (TODO)
ENV MODEL_NAME=resnet18
ENV MODEL_WEIGHTS_PATH=/app/models/best_model.pth
ENV DEVICE=cpu
ENV NUM_CLASSES=100
ENV CLASS_MAPPING_PATH=/app/models/class_id_to_name.json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health').raise_for_status()" || exit 1

# Run uvicorn server
CMD ["uvicorn", "plant_care_ai.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
