# Base image
FROM python:3.11-slim AS base

# App setup
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the local code to the container
COPY . .

# Set work directory and open the required port
EXPOSE 8501

# Run our service script
CMD ["streamlit", "run", "app.py"]
