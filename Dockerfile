# Dockerfile

FROM python:3.13-slim

# Use a neutral working directory like /code
WORKDIR /code

# Copy requirements first to leverage Docker layer caching
COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy your entire local 'app' folder into the /code directory
# This will create the correct /code/app/main.py structure
COPY ./app ./app

# This command now runs from /code and can correctly find the 'app' package
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
