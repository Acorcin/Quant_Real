# Quant_Real stack image: forecasting feeds + engine decision loop.
# One image, many services — docker-compose picks the command per service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf

WORKDIR /app

# CPU torch first (own index — PyPI's linux default bundles CUDA, ~6GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# code: forecasting pipeline, engine, vendored Kronos (trimmed by .dockerignore)
COPY forecasting/ forecasting/
COPY engine/ engine/
COPY vendor/ vendor/
COPY run_demo.py docker/bootstrap.py ./

# lock dir lives inside the container (PID namespaces don't cross)
ENV QUANT_LOCK_DIR=/tmp/locks

CMD ["python", "-c", "print('specify a service command via docker compose')"]
