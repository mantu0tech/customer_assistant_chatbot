FROM python:3.10-slim as build

WORKDIR /demo

RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.10-slim

WORKDIR /app

COPY --from=build /install /usr/local/

COPY . . 

EXPOSE 8501

CMD [ "streamlit", "run", "main.py"]


