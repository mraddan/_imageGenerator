FROM python:3.10.12-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 gcc g++ git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD bash entrypoint.sh