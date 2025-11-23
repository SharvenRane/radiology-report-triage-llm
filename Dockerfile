FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install all dependencies EXCEPT torch
RUN pip install --no-cache-dir -r requirements.txt

# Install sentence-transformers WITHOUT pulling torch
RUN pip install --no-deps sentence-transformers==2.7.0

COPY . .

CMD ["bash"]
