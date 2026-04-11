FROM python:3.12-slim
RUN pip install httpx --no-cache-dir
WORKDIR /app
COPY init.py .
CMD ["python", "init.py"]
