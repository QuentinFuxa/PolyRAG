FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
# Only install the client dependencies
RUN uv sync --frozen --only-group client

COPY .variables ./.variables
COPY src/ .
CMD ["streamlit", "run", "streamlit_app.py"]
