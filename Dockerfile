#FROM python:3.11.9-bookworm
FROM python:3.11.9

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

#EXPOSE 8000


