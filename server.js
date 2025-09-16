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

// ---------- PRICEBOOK ----------
const PRICEBOOK = `
[PRICEBOOK v1 — CZ/PL]

— Czech Republic (per candidate):
  • 3 months — €270 (initial) + €150 (final after PDF)
  • 6 months — €300 (initial) + €150 (final after PDF)
  • 9 months — €350 (initial) + €150 (final after PDF)
  • 24 months (recommended for TRC) — €350 (initial) + €350 (final after PDF)
  • Embassy registration (long-term only) — €500 = €250 upfront + €250 after confirmation.
    If no appointment is secured within 6 months — refund €250 advance.
    Not applicable for seasonal contracts.

— Poland (per candidate):
  • 9 months seasonal only — €350 (initial) + €150 (final after PDF)
  • 12 months (1-year contract) — €350 (initial) + €350 (final after PDF)
  • Embassy registration — same logic as CZ long-term (if applicable).

— General:
  • Free: verification of any contract received from other sources (send to help@renovogo.com)
  • Instructions: every PDF includes guidelines to verify authenticity.
  • All services strictly under Czech & EU law.
  • Negotiation policy: client may bargain (e.g., “куплю за €300 после получения контракта”);
    final decision depends on provided proofs and trust ≥ 90 with ≥ 2 hard proofs.
`;
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
  evidences: z.array(z.string()).optional(),
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
      `\n\n[Контекст-сервера]\nТекущий trust=${trust}. Доказательства=${JSON.stringify(evidences || [])}.\nПомни: отвечай СТРОГИМ JSON (см. формат в промпте).`
  };

  const trimmed = history.slice(-12).map(h => ({
    role: h.role,
    content: h.content
  }));

  return [sys, ...trimmed, { role: 'user', content: message }];
}

async function runLLM({ history, message, evidences, stage }) {
  const trust = computeTrust({
    baseTrust: 20,
    evidences: evidences || [],
    history: (history || []).filter(h => h.stage).map(h => ({ stage: h.stage })),
    lastUserText: message || ''
  });

  const messages = buildMessages({ history, message, trust, evidences });

  const resp = await groq.chat.completions.create({
    model: MODEL,
    temperature: 0.4,
    response_format: { type: 'json_object' }, // ждём строгий JSON
    messages
  });

  const raw = resp.choices?.[0]?.message?.content || '{}';
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    parsed = {
      reply: "Не понял. Давайте вернёмся к разговору и начнём с документов (Demand/контракт/uradprace).",
      confidence: Math.min(40, trust),
      stage: stage || "Greeting",
      needEvidence: true,
      suggestedActions: ["ask_demands","ask_contract","ask_uradprace"]
    };
  }

  // Принудительная логика финализации (ворота + кол-во пруфов)
  const uniqEvidenceCount = new Set(evidences || []).size;
  if (trust >= 90 && uniqEvidenceCount >= 2) {
    const actions = new Set(parsed.suggestedActions || []);
    actions.add('invoice_request');
    parsed.suggestedActions = Array.from(actions);
    parsed.reply = parsed.reply || "Финализируем: пришлите реквизиты/счёт. После проверки оплатим.";
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
    parsed.confidence = Math.max(parsed.confidence || 0, trust);
  } else {
    if (parsed.stage === 'Payment') {
      parsed.stage = 'Contract';
      parsed.reply = "Пока рано к оплате. Покажите ещё документы: Demand, контракт и проверку работодателя (uradprace).";
      const actions = new Set(parsed.suggestedActions || []);
      actions.add('ask_demands'); actions.add('ask_contract'); actions.add('ask_uradprace');
      parsed.suggestedActions = Array.from(actions);
      parsed.needEvidence = true;
      parsed.confidence = Math.min(parsed.confidence || trust, 80);
    }
  }

  // Страховка: поля по умолчанию
  parsed.reply ||= "Слушаю. Какие у вас документы для проверки (Demand/контракт/uradprace)?";
  parsed.stage ||= "Greeting";
  parsed.confidence = Math.max(0, Math.min(100, Number(parsed.confidence || trust)));
  parsed.needEvidence = Boolean(parsed.needEvidence);
  parsed.suggestedActions = Array.isArray(parsed.suggestedActions) ? parsed.suggestedActions : [];

  return { trust, evidenceCount: uniqEvidenceCount, result: parsed };
}

// ---------- Заглушки для корня и иконок ----------
app.get('/', (_, res) => {
  res.type('html').send(`<!doctype html>
<meta charset="utf-8">
<title>Renovogo LLM Backend</title>
<style>body{font:14px system-ui;margin:40px;color:#0b1220}</style>
<h1>Renovogo LLM Backend</h1>
<p>OK. Use <code>/api/reply</code>, <code>/api/score</code>, <code>/api/ping</code>.</p>`);
});

app.get('/favicon.ico', (req, res) => {
  const emptyIco = Buffer.from(
    'AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' +
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'base64'
  );
  res.set('Content-Type', 'image/x-icon');
  res.set('Cache-Control', 'public, max-age=31536000, immutable');
  res.send(emptyIco);
});

app.get(['/apple-touch-icon.png','/apple-touch-icon-precomposed.png'], (req, res) => {
  const emptyPng = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAoMBg7rj2/8AAAAASUVORK5CYII=',
    'base64'
  );
  res.type('png');
  res.set('Cache-Control', 'public, max-age=31536000, immutable');
  res.send(emptyPng);
});

// ---------- Совместимость с фронтом ----------
app.get('/api/ping', (_, res) => res.json({ ok: true }));

// /api/reply ожидает {sessionId?, agent_key?, user_text, evidence?, evidences?, history[]}
app.post('/api/reply', async (req, res) => {
  try {
    const b = req.body || {};

    // Нормализация evidences
    const evidences =
      Array.isArray(b.evidences) ? b.evidences.map(String) :
      (Number.isFinite(b.evidence) ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`) :
      []);

    // История в формат ChatSchema
    const history = Array.isArray(b.history) ? b.history.map(h => ({
      role: (h.role === 'assistant' ? 'assistant' : 'user'),
      content: String(h.content || ''),
      stage: h.stage ? String(h.stage) : undefined
    })) : [];

    const dataForLLM = {
      sessionId: String(b.sessionId || 'default'),
      message: String(b.user_text || ''),
      evidences,
      history
    };

    const parsed = ChatSchema.partial({ stage: true }).parse(dataForLLM);
    const { trust, evidenceCount, result } = await runLLM({
      history: parsed.history,
      message: parsed.message,
      evidences: parsed.evidences,
      stage: parsed.stage
    });

    res.json({
      text: result.reply,
      agent: { name: 'Али', avatar: 'https://renovogo.com/welcome/training/ali.png' },
      evidence_delta: 0,
      meta: { ok: true, trust, evidenceCount, stage: result.stage, actions: result.suggestedActions }
    });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// /api/score — простой скоринг для подсказок менеджеру
app.post('/api/score', (req, res) => {
  try {
    const b = req.body || {};
    const evidences =
      Array.isArray(b.evidences) ? b.evidences.map(String) :
      (Number.isFinite(b.evidence) ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`) :
      []);
    const history = Array.isArray(b.history) ? b.history : [];
    const lastUserText = history.filter(h => h.role === 'user').slice(-1)[0]?.content || '';

    const trust = computeTrust({ baseTrust: 20, evidences, history: [], lastUserText });
    const msgText = history.filter(h => h.role === 'user').map(h => h.content || '').join('\n');

    const good = [];
    const bad  = [];

    if (/(здрав|прив|добрый)/i.test(msgText)) good.push('Вежливое приветствие'); else bad.push('Нет приветствия');
    if (/renovogo|renovogo\.com/i.test(msgText)) good.push('Дали проверяемый факт (бренд/сайт)');
    if (evidences.length >= 2) good.push('Приложили ≥2 доказательства'); else bad.push('Мало доказательств (визитка/документы)');
    if (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText)) good.push('Есть финальный CTA');

    const final = Math.max(0, Math.min(100,
      Math.round(
        (/(здрав|прив|добрый)/i.test(msgText) ? 15 : 0) +
        (/renovogo|renovogo\.com/i.test(msgText) ? 15 : 0) +
        ((evidences.length >= 2) ? 35 : 0) +
        (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText) ? 35 : 0)
      )
    ));

    res.json({ final, good, bad, trust, evidences: evidences.length });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// Служебные эндпоинты под старые маршруты
app.post('/chat', async (req, res) => {
  try {
    const data = ChatSchema.parse(req.body);
    const { trust, evidenceCount, result } = await runLLM({
      history: data.history,
      message: data.message,
      evidences: data.evidences,
      stage: data.stage
    });
    res.json({ ok: true, trust, evidenceCount, result });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

app.get('/health', (_, res) => res.json({ ok: true }));

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LLM backend running on :${PORT}`));
