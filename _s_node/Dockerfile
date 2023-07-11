FROM library/python:3.6-stretch

WORKDIR /app


ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0

COPY requirements.txt /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "main_test.py"]
