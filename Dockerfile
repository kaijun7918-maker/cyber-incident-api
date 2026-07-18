FROM python:3.11-slim-bookworm

# Install Java 17 because PySpark requires a JVM.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Configure Java, Python and Spark environment variables.
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-17-openjdk-amd64/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYSPARK_PYTHON=python3 \
    SPARK_LOCAL_IP=127.0.0.1 \
    SPARK_LOCAL_HOSTNAME=localhost

# Set the application directory inside the Docker container.
WORKDIR /app

# Install the Python packages first to improve Docker build caching.
COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir \
    -r requirements.txt

# Copy the FastAPI code and trained Spark PipelineModel.
COPY main.py ./main.py
COPY model ./model

# Render uses port 10000 by default.
EXPOSE 10000

# Start the FastAPI service.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]