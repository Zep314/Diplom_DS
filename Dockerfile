#FROM python:3.11.9-bookworm
FROM python:3.11.9

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["/bin/bash","-c","cd /app && python main.py"]
