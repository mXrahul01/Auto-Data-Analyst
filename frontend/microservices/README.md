# React workspace

This directory contains the new microservices frontend. The existing `frontend/index.html`, `app.js`, and `style.css` are preserved as the earlier implementation.

Requires Node 22.12 or newer. From this directory: `npm ci`, then `npm run dev`. The dev server proxies `/api` to the local Compose stack at port 8080. For a production build, run `npm run build`. Docker serves that build through Nginx at http://localhost:8080.

This is a real service-status page. It does not display fake upload results, charts, or model metrics. Add feature-specific folders under `src/features` as the APIs become available. See the root microservices guide for the implementation sequence.
