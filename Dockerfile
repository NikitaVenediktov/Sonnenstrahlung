FROM python:3.10

WORKDIR /app

ADD requirements.txt .

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache

COPY app.py /app
COPY regressor.py /app
COPY predict.py /app
COPY data /app/data

EXPOSE 7070
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7070"]
