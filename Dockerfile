# Use a lightweight, official Python runtime
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the dependencies (and force it to ignore caches to keep the image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]