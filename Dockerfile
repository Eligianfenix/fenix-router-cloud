FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    FENIX_HOST=0.0.0.0

RUN useradd -m -u 1000 fenixuser
WORKDIR /app

COPY --chown=fenixuser:fenixuser inference_node.py master_dispatcher.py /app/

USER fenixuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request, os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 7860)}/health', timeout=3)" || exit 1

CMD ["python3", "/app/master_dispatcher.py"]
