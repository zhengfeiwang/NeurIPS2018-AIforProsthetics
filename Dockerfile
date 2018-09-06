FROM 192.168.1.40:5000/nips-deps:2.1.0
COPY baselines/ baselines
COPY nips/ nips
COPY logs/ logs
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"
