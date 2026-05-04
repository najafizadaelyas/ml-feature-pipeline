FROM apache/airflow:2.9.3-python3.11

USER root

# Install Java 17 (required for PySpark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

# Copy requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy pipeline source code
COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root config/ /opt/airflow/config/

ENV PIPELINE_CONFIG_PATH=/opt/airflow/config/pipeline_config.yaml
ENV PYTHONPATH=/opt/airflow
