FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --quiet --no-cache-dir -r requirements.txt

COPY . /app

ENV PORT 8080

CMD ["python3", "model.py"]

