FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY static /etc/nginx/static
COPY wait-for-openwebui.sh /docker-entrypoint.d/10-wait-for-openwebui.sh
RUN sed -i 's/\r$//' /docker-entrypoint.d/10-wait-for-openwebui.sh \
    && chmod +x /docker-entrypoint.d/10-wait-for-openwebui.sh
