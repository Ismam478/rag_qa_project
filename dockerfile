# Pulling Python Image
FROM python:3.11-slim

# Working Directory
WORKDIR /app

# Copying files into the docker Images
COPY . /app

# Installing dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Run the Application
CMD ["python", "main.py"]