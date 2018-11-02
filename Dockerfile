FROM login.local.inspir.ai:5000/nips-deps:2.1.0
COPY baselines/ baselines
COPY nips/ nips
COPY checkpoints/ checkpoints
COPY run.sh run.sh
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"
