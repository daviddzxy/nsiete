FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /nsiete && chmod -R a+rwx /nsiete/
WORKDIR /nsiete

EXPOSE 8888
EXPOSE 6006

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/nsiete --ip 0.0.0.0 --no-browser --allow-root"]
