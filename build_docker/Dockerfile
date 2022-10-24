#Use an official Python runtime as a parent image
FROM python:3.8-slim-buster 

#Set directory for translator
ENV PROJECT_ROOT=/opt/project
ENV APP_ROOT=/opt/project/Translator

#Set the working directory to /app
WORKDIR ${PROJECT_ROOT}
 
#Copy the current directory contents into the container at /app
COPY ./project .
RUN mv nltk_data /usr/share
RUN echo $(ls /usr/share)
 
#Install apt libraries
RUN apt update
RUN apt install -y sudo
RUN sudo apt-get install -y vim curl wget procps build-essential

#Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip 
RUN pip install --user -r requirements.txt

#Install nodejs & npm
RUN curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
RUN sudo apt-get install -y nodejs
WORKDIR ${APP_ROOT}
RUN echo $(node -v)
RUN echo $(npm version)
RUN npm install -g npm
RUN npm install npm@latest -g

#Install cuda-toolkit 11.2.0
WORKDIR ${PROJECT_ROOT}
RUN sudo apt-get install -y libxml2
RUN wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
