# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy only the required folders and files relative to the Dockerfile
# COPY data /app/data/
# COPY models /app/models/
COPY pages /app/pages/
COPY docker-requirements.txt /app/
COPY Home.py /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r docker-requirements.txt
RUN pip install streamlit

# Expose the port the app runs on
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "--server.address", "localhost", "--server.port", "8501", "Home.py"]

