FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y git curl libsm6 libxext6 libxrender-dev python3-tk

RUN pip3 install autolab_core opencv-python numpy flask
RUN pip3 install git+https://github.com/BerkeleyAutomation/perception.git
RUN pip3 install git+https://github.com/marctuscher/gqcnn.git
RUN pip3 uninstall -y tornado
RUN pip3 install tornado==5.1.1
RUN pip3 uninstall -y prompt_toolkit
RUN pip3 install prompt_toolkit==1.0.15
RUN pip3 install tensorflow
RUN pip3 uninstall -y scipy
RUN pip3 install scipy==1.2.1

EXPOSE 5000
RUN mkdir -p /usr/workdir/models
RUN mkdir -p /usr/workdir/practical
WORKDIR /usr/workdir
COPY models models
COPY practical practical

CMD ["python3", "practical/webserver/server.py"]