ARG PYTHON_VERSION="3.11"

FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip --no-cache-dir install --upgrade pip setuptools

COPY pyproject.toml .
RUN mkdir -p "src/plant_care_ai" && pip install --no-cache-dir ".[dev]" && rm -rf "src"

COPY src ./src
RUN pip install --no-cache-dir -e .

FROM python:${PYTHON_VERSION}-slim AS final

ENV USER=plant_user
ENV UID=1001
ENV GID=1001
ENV GROUP=plant_group
RUN addgroup --gid $GID $GROUP && adduser --uid $UID --gid $GID $USER

ENV HOME=/home/$USER
WORKDIR $HOME

COPY --from=builder --chown=$UID:$GID /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=$UID:$GID pyproject.toml .
COPY --chown=$UID:$GID src ./src

RUN pip install -e "."
USER $UID:$GID
