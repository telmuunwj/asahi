FROM pure/python:3.7-cuda10.0-runtime

WORKDIR /app

RUN apt-get update
# OpenCV dependencies
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY ./app/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy project
COPY ./app /app/

CMD exec gunicorn --bind 0.0.0.0:8080 --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 4 main:app