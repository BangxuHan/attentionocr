sudo docker run --runtime=nvidia  -it --name ocrtrain \
-v /home/kls/PycharmProjects/char/attentionocr:/home/kls/AttentionOCR-master \
-v /home/kls/PycharmProjects/char/dataset:/home/kls/numtrain \
tensorflow/tensorflow:1.15.0-gpu-py3
