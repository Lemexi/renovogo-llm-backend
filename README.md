# RenovoGo LLM Backend (Groq + Express)

Быстрый бэкенд для тренажёра менеджеров: Groq (Llama) + Node.js + Express.

## Локально (опционально)
1) `npm install`
2) Скопируй `.env.example` в `.env` и поставь свои значения:
   - `GROQ_API_KEY=<ваш_ключ>`
   - `ALLOWED_ORIGINS=https://ваш-домен`
   - `GROQ_MODEL=llama-3.1-8b-instant` (или любой из поддерживаемых Groq)
3) `npm start` → сервер на `http://localhost:3000`

## Деплой на Render
1) Создай репозиторий на GitHub и залей этот код.
2) На render.com → New → Web Service → подключи репозиторий.
3) Build Command: `npm install`  
   Start Command: `npm start`
4) В Environment добавь переменные:
   - `GROQ_API_KEY` (секретный!)
   - `ALLOWED_ORIGINS` (через запятую домены фронта)
   - `GROQ_MODEL` (опционально)
5) Открой `/health` → `{ ok: true }`.

## API
`POST /chat`
```json
{
  "sessionId": "demo-1",
  "message": "Здравствуйте",
  "stage": "Greeting",
  "evidences": ["business_card","contract"],
  "history": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
}
```

Ответ:
```json
{
  "ok": true,
  "trust": 95,
  "evidenceCount": 2,
  "result": {
    "reply": "Финализируем: пришлите реквизиты/кошелёк.",
    "confidence": 92,
    "stage": "Payment",
    "needEvidence": false,
    "suggestedActions": ["invoice_request"]
  }
}
```

## Безопасность
- Никогда не коммитьте `.env` и реальные ключи.
- Разрешайте CORS только для ваших доменов.