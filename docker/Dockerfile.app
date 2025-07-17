FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
RUN uv sync --frozen

COPY .variables ./.variables
COPY .streamlit ./.streamlit
COPY src/ .
COPY media ./media

CMD ["streamlit", "run", "streamlit-app.py"]
