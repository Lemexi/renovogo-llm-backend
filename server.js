// server.js
import 'dotenv/config';
import express from 'express';
import morgan from 'morgan';
import cors from 'cors';
import { Groq } from 'groq-sdk';
import { z } from 'zod';
import { SYSTEM_PROMPT } from './prompt.js';
import { computeTrust } from './trust.js';

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use(morgan('dev'));

// ---------- CORS ----------
const allowed = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (allowed.length === 0 || allowed.includes(origin)) return cb(null, true);
    cb(new Error('Not allowed by CORS: ' + origin));
  },
  credentials: false
}));

// ---------- Groq ----------
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';

// ---------- Schemas ----------
const ChatSchema = z.object({
  sessionId: z.string().min(1),
  message: z.string().min(1),
  stage: z.enum(["Greeting","Demand","Candidate","Contract","Payment","Closing"]).optional(),
  evidences: z.array(z.string()).optional(), // ["business_card","contract",...]
  history: z.array(z.object({
    role: z.enum(["user","assistant"]),
    content: z.string(),
    stage: z.string().optional()
  })).optional()
});

// ---------- Helpers ----------
function buildMessages({ history = [], message, trust, evidences }) {
  const sys = {
    role: 'system',
    content:
      SYSTEM_PROMPT +
      `\nТекущий trust=${trust}. Доказательства=${JSON.stringify(evidences || [])}.`
  };

  const trimmed = history.slice(-10).map(h => ({
    role: h.role,
    content: h.content
  }));

  return [sys, ...trimmed, { role: 'user', content: message }];
}

async function runLLM({ history, message, evidences }) {
  const trust = computeTrust({
    baseTrust: 30,
    evidences: evidences || [],
    history: (history || []).filter(h => h.stage).map(h => ({ stage: h.stage }))
  });

  const messages = buildMessages({ history, message, trust, evidences });

  const resp = await groq.chat.completions.create({
    model: MODEL,
    temperature: 0.5,
    response_format: { type: 'json_object' }, // строгий JSON
    messages
  });

  const raw = resp.choices?.[0]?.message?.content || '{}';
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    parsed = {
      reply: "Не понял. Давайте вернёмся к разговору.",
      confidence: 40,
      stage: "Greeting",
      needEvidence: true,
      suggestedActions: []
    };
  }

  // Принудительная логика финализации
  const evidenceCount = new Set(evidences || []).size;
  if (trust >= 90 && evidenceCount >= 2) {
    if (!Array.isArray(parsed.suggestedActions) || !parsed.suggestedActions.includes('invoice_request')) {
      parsed.suggestedActions = [...(parsed.suggestedActions || []), 'invoice_request'];
      parsed.reply = parsed.reply || "Финализируем: пришлите реквизиты/кошелёк.";
      parsed.stage = 'Payment';
    }
  } else {
    if (parsed.stage === 'Payment') {
      parsed.stage = 'Contract';
      parsed.reply = "Пока рано к оплате. Покажите ещё документы (контракт/пример визы/деманд).";
    }
  }

  return { trust, evidenceCount, result: parsed };
}

// ---------- Заглушки для корня и иконок (чтобы не было 404 в логах) ----------
app.get('/', (_, res) => {
  res.type('html').send(`<!doctype html>
<meta charset="utf-8">
<title>Renovogo LLM Backend</title>
<style>body{font:14px system-ui;margin:40px;color:#0b1220}</style>
<h1>Renovogo LLM Backend</h1>
<p>OK. Use <code>/api/reply</code>, <code>/api/score</code>, <code>/api/ping</code>.</p>`);
});

app.get('/favicon.ico', (req, res) => {
  // пустой 1×1 ICO
  const emptyIco = Buffer.from(
    'AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' +
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'base64'
  );
  res.set('Content-Type', 'image/x-icon');
  res.set('Cache-Control', 'public, max-age=31536000, immutable');
  res.send(emptyIco);
});

app.get(['/apple-touch-icon.png','/apple-touch-icon-precomposed.png'], (req, res) => {
  // прозрачный 1×1 PNG
  const emptyPng = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBg7rj2/8AAAAASUVORK5CYII=',
    'base64'
  );
  res.type('png');
  res.set('Cache-Control', 'public, max-age=31536000, immutable');
  res.send(emptyPng);
});

// ---------- Твои исходные роуты ----------
app.post('/chat', async (req, res) => {
  try {
    const data = ChatSchema.parse(req.body);
    const { trust, evidenceCount, result } = await runLLM({
      history: data.history,
      message: data.message,
      evidences: data.evidences
    });
    res.json({ ok: true, trust, evidenceCount, result });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

app.get('/health', (_, res) => res.json({ ok: true }));

// ---------- Совместимость с фронтом: /api/* ----------
app.get('/api/ping', (_, res) => res.json({ ok: true }));

// /api/reply ожидает {agent_key, user_text, evidence, evidences?, history[]}
app.post('/api/reply', async (req, res) => {
  try {
    const b = req.body || {};

    // Нормализация под ChatSchema
    const evidences =
      Array.isArray(b.evidences) ? b.evidences :
      (Number.isFinite(b.evidence) ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`) :
      []);

    const history = Array.isArray(b.history) ? b.history.map(h => ({
      role: (h.role === 'assistant' ? 'assistant' : 'user'),
      content: String(h.content || '')
    })) : [];

    const dataForLLM = {
      sessionId: b.sessionId || 'default',
      message: String(b.user_text || ''),
      evidences,
      history
    };

    // допускаем отсутствие stage
    const parsed = ChatSchema.partial({ stage: true }).parse(dataForLLM);

    const { trust, evidenceCount, result } = await runLLM({
      history: parsed.history,
      message: parsed.message,
      evidences: parsed.evidences
    });

    // Ответ для фронта
    res.json({
      text: result.reply || 'Слушаю вас.',
      agent: { name: 'Али', avatar: 'https://renovogo.com/welcome/assets/ali.png' },
      evidence_delta: 0,
      meta: { ok: true, trust, evidenceCount, stage: result.stage, actions: result.suggestedActions }
    });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// /api/score — простой скоринг
app.post('/api/score', (req, res) => {
  try {
    const b = req.body || {};
    const evidences =
      Array.isArray(b.evidences) ? b.evidences :
      (Number.isFinite(b.evidence) ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`) :
      []);
    const history = Array.isArray(b.history) ? b.history : [];

    const trust = computeTrust({ baseTrust: 30, evidences, history: [] });
    const msgText = history.filter(h => h.role === 'user').map(h => h.content || '').join('\n');

    const good = [];
    const bad  = [];

    if (/(здрав|прив|добрый)/i.test(msgText)) good.push('Вежливое приветствие'); else bad.push('Нет приветствия');
    if (/renovogo|renovogo\.com/i.test(msgText)) good.push('Дали проверяемый факт (бренд/сайт)'); else bad.push('Нет проверяемых фактов');
    if (evidences.length >= 2) good.push('Приложили 2+ доказательства'); else bad.push('Мало доказательств (визитка/документы)');
    if (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText)) good.push('Есть финальный CTA'); else bad.push('Нет финального CTA');

    const final = Math.max(0, Math.min(100,
      Math.round(
        (/(здрав|прив|добрый)/i.test(msgText) ? 20 : 0) +
        (/renovogo|renovogo\.com/i.test(msgText) ? 20 : 0) +
        ((evidences.length >= 2) ? 30 : 0) +
        (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText) ? 30 : 0)
      )
    ));

    res.json({ final, good, bad, trust, evidences: evidences.length });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LLM backend running on :${PORT}`));
