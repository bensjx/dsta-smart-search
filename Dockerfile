# For a GPU-enabled application
FROM tensorflow/tensorflow:2.1.0-gpu-py3

# Upgrade pip
RUN pip3 install --upgrade pip

# Install pip requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

# Set path
WORKDIR /app
ADD . /app

# Application can be accessed from localhost:80
EXPOSE 80

# Run "python main.py" which starts our application
CMD ["python", "main.py"]
