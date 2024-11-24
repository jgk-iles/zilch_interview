FROM python:3.11-slim

COPY app/ /app

COPY data/scored/ /app/data/

COPY requirements.txt /app

COPY models/ /app/models/

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py"]