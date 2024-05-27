FROM python:3.9-slim-bookworm
# RUN useradd -m -u 1000 user
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && \
    pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]