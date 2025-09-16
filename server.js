// server.js — RenovoGo LLM backend (stable memory, no duplicate asks, clear candidate logic)
// v2025-09-16-2

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
  .split(',').map(s => s.trim()).filter(Boolean);

app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (allowed.length === 0 || allowed.includes(origin)) return cb(null, true);
    cb(new Error('Not allowed by CORS: ' + origin));
  },
  credentials: false
}));

// ---------- Mini Rate Limit ----------
const rlStore = new Map(); // ip -> { count, reset }
function miniRateLimit(req, res, next){
  if (!req.path.startsWith('/api/')) return next();
  const ip = req.headers['x-forwarded-for']?.split(',')[0]?.trim() || req.ip || 'unknown';
  const now = Date.now();
  const rec = rlStore.get(ip) || { count: 0, reset: now + 60_000 };
  if (now > rec.reset) { rec.count = 0; rec.reset = now + 60_000; }
  rec.count += 1;
  rlStore.set(ip, rec);
  if (rec.count > 40) {
    res.set('Retry-After', Math.ceil((rec.reset - now)/1000));
    return res.status(429).json({ ok:false, error:'Too many requests, try again later.' });
  }
  next();
}
app.use(miniRateLimit);

// ---------- Groq ----------
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
const TEMPERATURE = Number(process.env.TEMPERATURE ?? 0.2);
const REPLY_MAX_TOKENS = Number(process.env.REPLY_MAX_TOKENS ?? 320);
const MAX_SENTENCES = Number(process.env.MAX_SENTENCES ?? 4);

// ---------- PRICEBOOK (в системный контекст) ----------
const PRICEBOOK = `
[PRICEBOOK v1 — CZ/PL]
— Czech Republic (per candidate):
  • 3m €270 + €150  • 6m €300 + €150  • 9m €350 + €150
  • 24m €350 + €350
  • Embassy reg (LT only): €500 = €250 + €250 (refund €250 if >6m no slot)
— Poland:
  • 9m seasonal €350 + €150  • 12m €350 + €350
— General: free verification; every PDF has verify guidelines; all under CZ/EU law.
`;

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

const LLMShape = z.object({
  reply: z.string().min(1),
  stage: z.enum(["Greeting","Demand","Candidate","Contract","Payment","Closing"]).optional(),
  confidence: z.number().min(0).max(100).optional(),
  needEvidence: z.boolean().optional(),
  suggestedActions: z.array(z.string()).optional()
});

// ---------- Utils ----------
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const extractFirstJsonObject = (s) => {
  const m = String(s||'').match(/\{[\s\S]*\}/);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
};
const sentSplit = (text) => String(text||'').split(/(?<=[.!?])\s+/).filter(Boolean);
const limitSentences = (text, max=4) => sentSplit(text).slice(0, max).join(' ').trim();
function logError(err, ctx=''){ console.error(`[${new Date().toISOString()}] ${ctx}:`, err?.stack || err); }
function forceMasculine(text){
  return String(text||'')
    .replace(/\bрада\b/gi, 'рад')
    .replace(/\bготова\b/gi, 'готов')
    .replace(/\bсмогла\b/gi, 'смог')
    .replace(/\bмогла\b/gi, 'мог');
}

// ---------- Stage actions ----------
const ACTION_WHITELIST = [
  "ask_demands",
  "ask_sample_contract",
  "ask_coop_contract",
  "ask_price_breakdown",
  "test_one_candidate",
  "invoice_request",
  "goodbye"
];
const ACTION_ORDER = new Map([
  ["ask_demands",1],
  ["ask_sample_contract",2],
  ["ask_coop_contract",3],
  ["ask_price_breakdown",4],
  ["test_one_candidate",5],
  ["invoice_request",6],
  ["goodbye",7]
]);
const normalizeActions = (arr) =>
  Array.from(new Set((Array.isArray(arr)?arr:[])
    .filter(a => ACTION_WHITELIST.includes(a))))
    .sort((a,b)=> (ACTION_ORDER.get(a)||99)-(ACTION_ORDER.get(b)||99));

// ---------- Консистентные ответы по слотам ----------
const REG_LONGTERM_MONTHS = 6;
const REG_SEASONAL_MONTHS = 3;
function registrationAnswer(){
  return `По долгосрочному — ${REG_LONGTERM_MONTHS} мес назад; по сезонному — ${REG_SEASONAL_MONTHS} мес назад. Очереди нестабильные, поэтому сначала проверю ваш Demand Letter и контракт.`;
}

// ---------- Микро-состояние сессии ----------
const sessionState = new Map();
function getState(sid='default'){
  if (!sessionState.has(sid)) {
    sessionState.set(sid, {
      lastReply: '',
      lastActions: [],
      seenEvidences: new Set() // устойчивое хранение доказательств
    });
  }
  return sessionState.get(sid);
}

// ---------- Сборка сообщений в LLM ----------
function buildMessages({ history = [], message, trust, evidences }) {
  const sys = {
    role: 'system',
    content:
      SYSTEM_PROMPT +
      `\n\n[Контекст]\ntrust=${trust}; evidences=${JSON.stringify(evidences || [])}\n${PRICEBOOK}\n` +
      `Отвечай СТРОГО одним JSON-объектом (см. формат). Будь кратким (до ${MAX_SENTENCES} предложений).`
  };
  const trimmed = (history||[]).slice(-12).map(h => ({
    role: h.role, content: h.content
  }));
  return [sys, ...trimmed, { role: 'user', content: message }];
}

// ---------- Обёртка с ретраями ----------
async function createChatWithRetry(payload, tries = 2) {
  let lastErr;
  while (tries--) {
    try {
      return await groq.chat.completions.create(payload);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

// ---------- Пост-правила (учёт памяти, кандидаты, финализация) ----------
function postRules({ parsed, trust, evidences, history, userText, sid }) {
  const S = getState(sid);

  // 1) Сливаем входящие доказательства с памятью сессии
  const inc = new Set(evidences || []);
  const ev = new Set([...(S.seenEvidences || new Set()), ...inc]); // объединённый набор
  const isNew = k => inc.has(k) && !S.seenEvidences.has(k);

  let reply = String(parsed.reply || '').trim();

  // Быстрые точные ответы
  if (/(как.*зовут|вас зовут|ваше имя|who are you)/i.test(userText)) {
    reply = 'Меня зовут Али.';
    parsed.stage ??= 'Greeting';
  }
  // Регистрации/слоты — консистентный текст
  if (/(когда|последн).*(регистрир|записыва)|слот|очеред/i.test(userText)) {
    reply = registrationAnswer();
    parsed.stage = 'Demand';
    parsed.needEvidence = true;
    parsed.suggestedActions = ['ask_demands','ask_coop_contract'];
  }

  // Вопрос про количество кандидатов — отвечаем предметно
  const askCandidates = /(сколько.*кандидат|кандидат(ов)?\s*(есть|готов|подад)|сколько человек будем подавать)/i.test(userText);
  if (askCandidates) {
    if (ev.has('demand_letter') && ev.has('coop_contract_pdf')) {
      reply = 'Готов подать 1–2 кандидата на старт в Чехию. По остальным согласуем после финализации и слотов.';
      parsed.stage = parsed.stage === 'Payment' ? 'Payment' : 'Candidate';
      parsed.needEvidence = false;
      const set = new Set(parsed.suggestedActions || []);
      set.add('invoice_request');
      parsed.suggestedActions = Array.from(set);
    } else {
      const miss = [];
      if (!ev.has('demand_letter')) miss.push('Demand Letter');
      if (!ev.has('coop_contract_pdf')) miss.push('полный контракт о сотрудничестве');
      reply = `Смогу подтвердить точное число сразу после ${miss.join(' и ')}. Сейчас ориентир — 1–2 кандидата на старт.`;
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      parsed.suggestedActions = ['ask_demands','ask_coop_contract'];
    }
  }

  // Не инициировать оплату при низком доверии
  if (parsed.stage === 'Payment' && trust < 90) {
    reply = 'К оплате перейдём после проверки документов и согласования контракта.';
    parsed.stage = 'Contract';
    parsed.needEvidence = true;
    parsed.suggestedActions = ['ask_demands','ask_coop_contract','ask_price_breakdown'];
  }

  // Признание новых доказательств (ACK) + корректные запросы
  const acks = [];
  if (isNew('demand_letter')) {
    acks.push('Спасибо за Demand Letter.');
    if (!ev.has('coop_contract_pdf')) {
      acks.push('Теперь пришлите полноценный контракт о сотрудничестве (полная версия, с печатью/подписью).');
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      parsed.suggestedActions = ['ask_coop_contract'];
    }
  }
  if (isNew('sample_contract_pdf') && !ev.has('coop_contract_pdf')) {
    acks.push('Пример контракта получен. Для проверки нужен полный контракт о сотрудничестве.');
    parsed.stage = 'Contract';
    parsed.needEvidence = true;
    parsed.suggestedActions = ['ask_coop_contract'];
  }
  if (isNew('coop_contract_pdf')) {
    acks.push('Контракт о сотрудничестве получен.');
    parsed.stage = 'Contract';
    parsed.needEvidence = false;
    const set = new Set(parsed.suggestedActions || []);
    set.add('ask_price_breakdown');
    parsed.suggestedActions = Array.from(set);
  }
  if (isNew('visa_sample')) acks.push('Пример визы получен.');
  if (isNew('presentation')) acks.push('Презентация получена.');
  if (isNew('video')) acks.push('Видео получено.');

  if (acks.length) {
    reply = (reply ? (limitSentences(reply, MAX_SENTENCES) + ' ') : '') + acks.join(' ');
  }

  // Правка лишних «реквизитов работодателя» при наличии Demand
  if (ev.has('demand_letter') && /реквизит/i.test(reply)) {
    reply = reply.replace(/[^.?!]*реквизит[^.?!]*работодател[^.?!]*[.?!]/gi, '').trim();
    if (!ev.has('coop_contract_pdf')) {
      reply += (reply ? ' ' : '') + 'Реквизиты у меня уже есть из Demand. Нужен полный контракт о сотрудничестве.';
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      parsed.suggestedActions = ['ask_coop_contract'];
    }
  }

  // Вопрос про способ оплаты — жёстко «банк»
  if (/(банк|банковск|crypto|крипто|usdt|криптовалют)/i.test(userText) && /оплат|плат[её]ж|инвойс/i.test(userText)) {
    reply = 'Банковский инвойс. Криптовалюту не принимаем.';
    const set = new Set(parsed.suggestedActions || []);
    set.add('invoice_request');
    parsed.suggestedActions = Array.from(set);
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  // Убираем клише и правим род
  reply = forceMasculine(limitSentences(
    reply.replace(/(?:оставьте заявку.*?|мы предлагаем широкий спектр услуг)/gi, '').trim(),
    MAX_SENTENCES
  ));

  // Анти-луп
  if (reply && S.lastReply && reply.toLowerCase() === S.lastReply.toLowerCase()) {
    reply = 'Коротко: документы принял. Готов подать 1–2 кандидата и перейти к инвойсу.';
    parsed.stage = 'Payment';
    const set = new Set(parsed.suggestedActions || []);
    set.add('invoice_request');
    parsed.suggestedActions = Array.from(set);
  }
  S.lastReply = reply;

  // Ворота финализации — требуем контракт
  const uniqEvidenceCount = ev.size;
  const hasCoop = ev.has('coop_contract_pdf');
  const gatesOk = (trust >= 90 && uniqEvidenceCount >= 2 && hasCoop);
  if (gatesOk) {
    const set = new Set(parsed.suggestedActions || []);
    set.add('invoice_request');
    parsed.suggestedActions = Array.from(set);
    if (!/инвойс|сч[её]т|реквизит/i.test(reply)) {
      reply += (reply ? ' ' : '') + 'Готов перейти к финализации — пришлите реквизиты для инвойса.';
    }
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  parsed.reply = reply.trim();
  parsed.suggestedActions = normalizeActions(parsed.suggestedActions);

  // Если просим документы — stage не ниже Demand
  if (/(demand|контракт|документ|полный контракт|сотрудничеств)/i.test(parsed.reply) && (!parsed.stage || parsed.stage === 'Greeting')) {
    parsed.stage = 'Demand';
  }

  // 2) Сохраняем объединённый набор в сессию (память)
  S.seenEvidences = ev;

  return parsed;
}

// ---------- Вызов LLM ----------
async function runLLM({ history, message, evidences, stage, sessionId='default' }) {
  const trust = computeTrust({
    baseTrust: 20,
    evidences: evidences || [],
    history: history || [],
    lastUserText: message || ''
  });

  const messages = buildMessages({ history, message, trust, evidences });

  const resp = await createChatWithRetry({
    model: MODEL,
    temperature: TEMPERATURE,
    top_p: 0.9,
    frequency_penalty: 0.4,
    presence_penalty: 0.0,
    max_tokens: REPLY_MAX_TOKENS,
    response_format: { type: 'json_object' },
    messages
  });

  const raw = resp?.choices?.[0]?.message?.content || '{}';
  const json = extractFirstJsonObject(raw);

  let parsed;
  try { parsed = LLMShape.parse(json); } catch { parsed = null; }

  if (!parsed) {
    const fb = stage === 'Payment'
      ? 'К оплате перейдём после проверки документов.'
      : 'Нужны Demand Letter и контракт о сотрудничестве. После проверки обсудим сроки и цену.';
    parsed = {
      reply: fb,
      confidence: clamp(trust, 0, 60),
      stage: stage || 'Demand',
      needEvidence: true,
      suggestedActions: ['ask_demands','ask_coop_contract']
    };
  }

  // Страховки типов
  parsed.reply = String(parsed.reply || '').trim();
  parsed.stage = String(parsed.stage || stage || "Greeting");
  parsed.confidence = clamp(Number(parsed.confidence ?? trust), 0, 100);
  parsed.needEvidence = Boolean(parsed.needEvidence);
  parsed.suggestedActions = Array.isArray(parsed.suggestedActions) ? parsed.suggestedActions : [];

  // Пост-правила
  const sid = sessionId || 'default';
  parsed = postRules({
    parsed,
    trust,
    evidences,
    history,
    userText: message || '',
    sid
  });

  const uniqEvidenceCount = new Set(evidences || []).size;
  return { trust, evidenceCount: uniqEvidenceCount, result: parsed };
}

// ---------- Root & assets ----------
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

function sanitizeHistory(arr){
  return Array.isArray(arr) ? arr.slice(-50).map(h => ({
    role: (h.role === 'assistant' ? 'assistant' : 'user'),
    content: String(h.content || '').replace(/<[^>]+>/g, ''),
    stage: h.stage ? String(h.stage) : undefined
  })) : [];
}

// Нормализация ключей доказательств из фронта
function normalizeEvidenceKey(k){
  const key = String(k || '').toLowerCase().trim();
  const map = new Map([
    // Demand
    ['demand','demand_letter'], ['demandletter','demand_letter'], ['деманд','demand_letter'],
    // Contracts
    ['sample','sample_contract_pdf'], ['sample_contract','sample_contract_pdf'],
    ['contract_sample','sample_contract_pdf'], ['пример_контракта','sample_contract_pdf'],
    ['contract_pdf','coop_contract_pdf'], // важно для вашего фронта
    ['contract','coop_contract_pdf'], ['contractpdf','coop_contract_pdf'], ['договор','coop_contract_pdf'],
    ['coop_contract','coop_contract_pdf'], ['full_contract','coop_contract_pdf'],
    ['контракт_о_сотрудничестве','coop_contract_pdf'],
    // Other proofs
    ['visa','visa_sample'], ['visa_scan','visa_sample'], ['пример_визы','visa_sample'], ['visa_sample','visa_sample'],
    ['card','business_card'], ['визитка','business_card'],
    ['site','website'], ['сайт','website'], ['website','website'],
    ['reviews','reviews'], ['отзывы','reviews'],
    ['registry','registry_proof'], ['uradprace','registry_proof'], ['регистрация','registry_proof'],
    ['presentation','presentation'], ['deck','presentation'], ['video','video'], ['youtube','video']
  ]);
  return map.get(key) || key;
}

// /api/reply — основной
app.post('/api/reply', async (req, res) => {
  try {
    const b = req.body || {};

    const rawMessage = String(b.user_text ?? b.message ?? '').trim();
    if (!rawMessage || rawMessage.length > 2000) {
      return res.status(400).json({ ok: false, error: 'Invalid message length' });
    }

    const evidences = Array.isArray(b.evidences)
      ? [...new Set(b.evidences.map(normalizeEvidenceKey).filter(Boolean))]
      : (Number.isFinite(b.evidence)
          ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`)
          : []);

    const history = sanitizeHistory(b.history);

    const { trust, evidenceCount, result } = await runLLM({
      history,
      message: rawMessage,
      evidences,
      stage: b.stage,
      sessionId: String(b.sessionId || 'default')
    });

    res.json({
      text: result.reply,
      agent: { name: 'Али', avatar: 'https://renovogo.com/welcome/training/ali.png' },
      evidence_delta: 0,
      meta: {
        ok: true,
        trust,
        evidenceCount,
        stage: result.stage,
        actions: normalizeActions(result.suggestedActions)
      }
    });
  } catch (e) {
    logError(e, '/api/reply');
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// /api/score — подсказки менеджеру
app.post('/api/score', (req, res) => {
  try {
    const b = req.body || {};
    const evidences = Array.isArray(b.evidences)
      ? [...new Set(b.evidences.map(normalizeEvidenceKey).filter(Boolean))]
      : (Number.isFinite(b.evidence)
          ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`)
          : []);
    const history = sanitizeHistory(b.history);
    const lastUserText = history.filter(h => h.role === 'user').slice(-1)[0]?.content || '';

    const trust = computeTrust({ baseTrust: 20, evidences, history, lastUserText });

    const msgText = history.filter(h => h.role === 'user').map(h => h.content || '').join('\n');

    const good = [];
    const bad  = [];

    if (/(здрав|прив|добрый)/i.test(msgText)) good.push('Вежливое приветствие'); else bad.push('Нет приветствия');
    if (/renovogo|renovogo\.com/i.test(msgText)) good.push('Дали проверяемый факт');
    if (evidences.length >= 2) good.push('Приложили ≥2 доказательства'); else bad.push('Мало доказательств');
    if (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText)) good.push('Есть финальный CTA');

    const final = clamp(Math.round(
      (/(здрав|прив|добрый)/i.test(msgText) ? 15 : 0) +
      (/renovogo|renovogo\.com/i.test(msgText) ? 15 : 0) +
      ((evidences.length >= 2) ? 35 : 0) +
      (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText) ? 35 : 0)
    ), 0, 100);

    if (trust < 80) bad.push('Для предметного обсуждения добавьте документы (Demand/Contract/Registry).');

    res.json({ final, good, bad, trust, evidences: evidences.length });
  } catch (e) {
    logError(e, '/api/score');
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// Совместимость со старым роутом
app.post('/chat', async (req, res) => {
  try {
    const data = ChatSchema.parse({
      sessionId: String(req.body?.sessionId || 'default'),
      message: String(req.body?.message || '').trim(),
      stage: req.body?.stage,
      evidences: Array.isArray(req.body?.evidences) ? req.body.evidences.map(normalizeEvidenceKey) : [],
      history: sanitizeHistory(req.body?.history)
    });
    const { trust, evidenceCount, result } = await runLLM({
      history: data.history,
      message: data.message,
      evidences: data.evidences,
      stage: data.stage,
      sessionId: data.sessionId
    });
    res.json({ ok: true, trust, evidenceCount, result });
  } catch (e) {
    logError(e, '/chat');
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

app.get('/health', (_, res) => res.json({ ok: true }));

// ---------- Start ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LLM backend running on :${PORT}`));
