# Use Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model folder explicitly
COPY model /app/model

# Copy all other files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app on port 8501, listen on all interfaces
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
