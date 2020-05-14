FROM python:3

ADD requirements.txt /
RUN pip install -r requirements.txt

COPY . /usr/src/gbn
RUN pip install /usr/src/gbn

ENTRYPOINT ["gbn"]
