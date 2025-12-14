FROM python:3-alpine

ENV PYTHONUNBUFFERED 1

WORKDIR /workroot

RUN apk add --no-cache uv

COPY backend/pyproject.toml .

RUN uv sync

RUN adduser -D runner

USER runner

CMD ["uv", "run", "."]
