
FROM python:3.7-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

CMD ["python", "run.app", "--doubleDQN", "--greedy", "UI"]