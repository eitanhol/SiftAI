# Use a base image with both Node.js and Python installed
FROM python:3.13-bullseye

# Install Node.js
RUN apt-get update && apt-get install -y nodejs npm

# Install Doppler CLI
RUN curl -L --fail https://cli.doppler.com/install.sh | sh

# Set the working directory in the container
WORKDIR /app

# Copy the Node.js project files and install dependencies
COPY DataAnalyzer/package.json ./DataAnalyzer/
COPY DataAnalyzer/chatbot.js ./DataAnalyzer/

# Copy the Python project files and install dependencies
COPY DataAnalyzer/requirements.txt ./DataAnalyzer/
COPY DataAnalyzer/main.py ./DataAnalyzer/
COPY DataAnalyzer/Procfile ./DataAnalyzer/
# ... copy other necessary files from DataAnalyzer/

# Set the working directory to the app folder and install dependencies
WORKDIR /app/DataAnalyzer
RUN npm install
RUN pip install -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8000

# Run the application using the Procfile command
CMD doppler run -- gunicorn main:app --workers=4 --bind=0.0.0.0:$PORT