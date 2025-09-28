# Dockerfile

FROM python:3.13-slim

# Use a neutral working directory like /code
WORKDIR /code

# Copy requirements first to leverage Docker layer caching
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your application code
COPY ./app ./app

# --- ADD THIS LINE ---
# Copy the frontend files into the image
COPY ./frontend ./frontend

# This command now runs from /code and can correctly find the 'app' package
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
