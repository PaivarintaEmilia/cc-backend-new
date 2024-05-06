# syntax=docker/dockerfile:1

# Start your image with a node base image
FROM python:3.12-bullseye

# The /app directory should act as the main application directory
WORKDIR /app

# Copy local directories to the current local directory of our docker image (/app)
COPY . /app

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN pip install --no-cache-dir --progress-bar off -r requirements.txt

# Käytä samaa mitä oma ohjelma
EXPOSE 8080

# Start the app using serve command
CMD ["python", "app.py"]