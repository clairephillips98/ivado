FROM python:slim
WORKDIR /app
COPY . .
COPY UNdata_Export_20230308_220221493.csv .
RUN pip install --no-cache-dir -r requirements.txt

ENV museum='none'
ENV un_data='none'
ENV regression='none'

CMD python main.py $museum $un_data $regression