FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --default-timeout=100 --retries 10 --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 5001

CMD ["python", "-m", "app.app"]

