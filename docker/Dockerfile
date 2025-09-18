FROM python:3.12-slim

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV \
    && $VIRTUAL_ENV/bin/pip install --upgrade pip setuptools wheel
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt README.md LICENSE /app/
COPY src /app/src
COPY configs /app/configs
COPY tests /app/tests
COPY AGENTS.md /app/AGENTS.md

RUN pip install -e .[dev]

ENTRYPOINT ["hei-seti"]
CMD ["--help"]
