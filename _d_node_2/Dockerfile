FROM python:3.7

WORKDIR /app

RUN touch input.txt
RUN touch output.txt

COPY requirements.txt /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]
