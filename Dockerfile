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
COPY params.yaml /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r docker-requirements.txt



# Run app.py when the container launches
CMD ["streamlit", "run", "Home.py"]

