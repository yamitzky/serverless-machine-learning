FROM continuumio/miniconda

RUN mkdir -p /usr/src/app

WORKDIR /usr/src/app

COPY download_corpus.sh /usr/src/app/
RUN sh download_corpus.sh

COPY conda-requirements.txt /usr/src/app/

RUN conda create -y -n deploy --file conda-requirements.txt
# 関連ライブラリは、/opt/conda/envs/deploy/lib/python2.7/site-packages に吐き出される

COPY . /usr/src/app/

# 学習し、モデルの吐き出しを行う
RUN python gen_corpus.py \
      && /bin/bash -c "source activate deploy && python train.py"

RUN mkdir -p build/lib \
      && cp main.py model.pkl build/ \
      && cp -r /opt/conda/envs/deploy/lib/python2.7/site-packages/* build/ \
      && cp /opt/conda/envs/deploy/lib/libopenblas* /opt/conda/envs/deploy/lib/libgfortran* build/lib/
