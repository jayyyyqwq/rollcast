# syntax=docker/dockerfile:1
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY . .
RUN uv sync --locked --no-dev
RUN useradd -u 10001 app && mkdir /data && chown app /data
USER app
WORKDIR /data
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["rollcast"]
CMD ["--help"]
