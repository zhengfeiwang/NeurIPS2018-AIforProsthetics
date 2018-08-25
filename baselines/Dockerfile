# The deploy Docker image build a self-contained Ray instance suitable
# for end users.

FROM 192.168.1.40:5000/baselines-deps:latest
COPY baselines/ baselines
COPY dev/ dev
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"
