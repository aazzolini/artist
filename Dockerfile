FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shop ./shop
COPY products ./products
# Catalog and piece files live on the persistent disk; the shipped products seed it on first start.
ENV DB_PATH=/data/orders.db PRODUCTS_DIR=/data/products SEED_PRODUCTS_DIR=/app/products
VOLUME /data
EXPOSE 8000
# Hosts like Render pass the port to listen on in $PORT.
CMD ["sh", "-c", "exec uvicorn shop.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
