# Start with Python 3.10 (like a base ingredient)
FROM python:3.10-slim

# Create a folder inside the container called /app
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Tell Docker: "My app uses port 8501"
EXPOSE 8501

# When container starts, run this command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
