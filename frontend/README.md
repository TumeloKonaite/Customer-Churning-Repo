# Churn Insight frontend

React, TypeScript, and Vite client for the existing Customer Churn FastAPI service. The application does not persist customer inputs, call protected monitoring endpoints, or contain API credentials.

## Prerequisites

- Node.js 20 or newer
- npm 10 or newer
- A reachable FastAPI deployment

## Local development

```bash
cp .env.example .env.local
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in `.env.local` to the API origin, without a trailing path. Values prefixed with `VITE_` are public in the browser bundle, so this variable must contain only the non-secret API URL.

The backend must include `http://localhost:5173` in `FRONTEND_ALLOWED_ORIGINS`. Development and test backend environments add the standard local Vite origins automatically.

## Test and build

```bash
npm test
npm run build
npm run preview
```

The static production output is written to `dist/`. Test direct navigation to `/`, `/predict`, and `/batch` through the preview or deployed host.

## Deploy

Set the project root/base directory to `frontend`, the build command to `npm run build`, and the output directory to `dist`.

- Vercel: import the repository and use the included `vercel.json` rewrite.
- Netlify: use the included `netlify.toml`; `public/_redirects` is also copied into the build.
- Cloudflare Pages: use `npm run build` and `dist`; the included `_redirects` file provides the SPA fallback.

Configure the production host with:

```env
VITE_API_BASE_URL=https://tumelokonaitedev--customer-churn-backend-fastapi-app.modal.run
```

Then configure the backend with an exact allowlist and redeploy it:

```env
APP_ENV=production
FRONTEND_ALLOWED_ORIGINS=https://your-production-frontend.example.com
```

For multiple approved sites, separate exact origins with commas. Wildcards are rejected. After both deployments, open every route directly, confirm the live health metadata, and perform one non-sensitive single and JSON batch prediction.

## Production deployment

- Production URL: https://customer-churning-repo.vercel.app
- API origin: https://tumelokonaitedev--customer-churn-backend-fastapi-app.modal.run
- Successful production smoke test: 2026-08-24

The production smoke test confirmed direct navigation and refresh for `/`, `/predict`, and `/batch`; healthy model metadata; exact-origin CORS; one single-customer prediction; one JSON batch prediction; and no browser console, mixed-content, CORS, or failed API-request errors.
