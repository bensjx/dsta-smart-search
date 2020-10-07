FROM tensorflow/tensorflow:2.1.0-gpu-py3

# For more information, please refer to https://aka.ms/vscode-docker-python
# FROM python:3.8-slim-buster
RUN pip3 install --upgrade pip

# Install pip requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
ADD . /app

EXPOSE 80

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]
