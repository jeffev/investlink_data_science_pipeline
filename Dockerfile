FROM python:3.11-slim

WORKDIR /app

COPY requirements-predict.txt .
RUN pip install --no-cache-dir -r requirements-predict.txt

COPY . .

CMD ["python", "pipeline.py", "--predict"]
