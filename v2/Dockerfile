FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY v2/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --system --create-home --uid 10001 app

COPY --chown=app:app v2 /app/v2
COPY --chown=app:app guardrails /app/guardrails

USER app
EXPOSE 8000
CMD ["python", "-m", "v2.mcp_server"]

