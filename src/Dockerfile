FROM python:3.10

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

copy ./routers/* /app/routers/

copy ./templates/* /app/templates/

copy ./*.py /app/

copy ./unet_model.h5 /app/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]