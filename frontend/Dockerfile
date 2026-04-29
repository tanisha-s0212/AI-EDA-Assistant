FROM node:22-bullseye-slim

WORKDIR /app

COPY package*.json ./
COPY start.sh /app/start.sh
COPY public /app/public

RUN chmod +x /app/start.sh

ENV HOSTNAME="0.0.0.0"
EXPOSE 3000

CMD ["/app/start.sh"]
