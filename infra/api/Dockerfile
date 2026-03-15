FROM python:3-alpine

ENV PYTHONUNBUFFERED 1

WORKDIR /workroot

RUN apk add --no-cache uv && \
    adduser -D runner && \
    chown runner:runner /workroot

USER runner

COPY --chown=runner backend/pyproject.toml .

RUN uv sync

CMD ["uv", "run", "."]
