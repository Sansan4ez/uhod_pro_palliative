# syntax=docker/dockerfile:1
FROM docker.io/continuumio/miniconda3:24.3.0-0
WORKDIR /
COPY ./flow/requirements.txt /flow/requirements.txt
# create conda environment
RUN conda create -n promptflow-serve python=3.9.18 pip=23.0.1 -q -y && \
    conda run -n promptflow-serve pip install -r /flow/requirements.txt && \
    conda run -n promptflow-serve pip install keyrings.alt && \
    conda run -n promptflow-serve pip install gunicorn==20.1.0 && \
    conda run -n promptflow-serve pip cache purge && \
    conda clean -a -y
COPY ./flow /flow
COPY ./baze /baze
RUN apt-get update && apt-get install --no-install-recommends -y runit
EXPOSE 8080
COPY ./connections/* /connections/
# reset runsvdir
RUN rm -rf /var/runit
COPY ./runit /var/runit
# grant permission
RUN chmod -R +x /var/runit
COPY ./start.sh /
CMD ["bash", "./start.sh"]