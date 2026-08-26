ARG VARIANT=3.14-slim
FROM python:${VARIANT}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app

COPY . .

EXPOSE 8501
EXPOSE 4213

# ENTRYPOINT
ENTRYPOINT ["bash"]