FROM node:22-alpine AS build
WORKDIR /web
COPY frontend/microservices/package*.json ./
RUN npm ci
COPY frontend/microservices/ ./
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.28-alpine
COPY --from=build /web/dist /usr/share/nginx/html
COPY infrastructure/nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
