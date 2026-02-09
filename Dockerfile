FROM python:3.11

WORKDIR /app

RUN pip install --upgrade pip && \
    pip config set global.timeout 1000 && \
    pip config set global.retries 10

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]