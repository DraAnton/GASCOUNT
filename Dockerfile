FROM python:latest

WORKDIR /app
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt 

COPY . .
#COPY /data/plots/existing_plots.json /data/plots/existing_plots.json
ENTRYPOINT ["/bin/bash", "starter.sh"]
