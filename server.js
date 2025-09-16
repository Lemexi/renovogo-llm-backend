// server.js
import 'dotenv/config';
import express from 'express';
import morgan from 'morgan';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
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

// ---------- Rate limit ----------
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 мин
  max: 30,             // 30 req/IP/мин
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

// ---------- Groq ----------
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';

// ---------- PRICEBOOK (в системный контекст, не в ответ) ----------
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

// Валидация ответа LLM (жёсткая)
const LLMShape = z.object({
  reply: z.string().min(1),
  stage: z.enum(["Greeting","Demand","Candidate","Contract","Payment","Closing"]).optional(),
  confidence: z.number().min(0).max(100).optional(),
  needEvidence: z.boolean().optional(),
  suggestedActions: z.array(z.string()).optional()
});

// ---------- Utils ----------
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const pick  = (arr) => arr[Math.floor(Math.random() * arr.length)];
const trimToSentences = (text, max = 6) => {
  const parts = String(text || '').split(/(?<=[.!?])\s+/).filter(Boolean);
  return parts.slice(0, max).join(' ').trim();
};
const extractFirstJsonObject = (s) => {
  const m = String(s||'').match(/\{[\s\S]*\}/);
  if (!m) return null;
  try { return JSON.parse(m[0]); } catch { return null; }
};
const lastAssistantReplyFromHistory = (history=[]) =>
  String((history || []).filter(h => h.role==='assistant').slice(-1)[0]?.content || '').trim();

function logError(err, ctx=''){
  console.error(`[${new Date().toISOString()}] ${ctx}:`, err?.stack || err);
}

// --- actions normalize (урезали до нужных и согласовали с prompt.js) ---
const ACTION_WHITELIST = [
  "invoice_request","ask_demands","ask_contract",
  "ask_price_breakdown","test_one_candidate","goodbye"
];
const ACTION_ORDER = new Map([
  ["ask_demands",1],["ask_contract",2],
  ["ask_price_breakdown",3],["test_one_candidate",4],
  ["invoice_request",5],["goodbye",6]
]);
const normalizeActions = (arr) =>
  Array.from(
    new Set((Array.isArray(arr)?arr:[]).filter(a => ACTION_WHITELIST.includes(a)))
  ).sort((a,b)=> (ACTION_ORDER.get(a)||99)-(ACTION_ORDER.get(b)||99));

// — человечные фразы
const FALLBACK_HUMAN = [
  "Не понял вас. О чём речь?",
  "Секунду, вы что имеете в виду?",
  "Поясните, я не уловил мысль.",
  "Не совсем понял. Конкретнее можно?",
  "О чём говорим? Дайте контекст.",
  "Подождите, не понял. Сформулируйте проще.",
  "Сейчас не понял. Что вы хотите конкретно?",
  "Я выпал из контекста. Повторите коротко."
];

const ASK_DOCS_HUMAN = [
  "Сначала пришлите запрос по вакансии (Demand) — я проверю.",
  "Нужен Demand или ваш договор о сотрудничестве. Тогда обсудим дальше.",
  "Отправьте документы по вакансии и ваш контракт — без бумаг не двигаюсь.",
  "Мне нужны конкретные документы: Demand Letter и контракт. Потом пойдём дальше.",
  "Давайте по-взрослому: документы — затем обсуждения."
];

const TOO_EARLY_PAY_HUMAN = [
  "Какая оплата? Мы ещё ничего не проверили.",
  "Оплата обсуждается после проверки документов.",
  "Пока рано говорить про деньги. Сначала Demand и контракт.",
  "Платить за воздух не буду. Документы — вперёд.",
  "Не торопите события: проверки → потом инвойс."
];

// тематические подсказки
const CRYPTO_SKEPTIC = [
  "Крипта — риск. Предпочитаю счёт-фактуру и банковский перевод.",
  "Крипта? Не лучший знак. Давайте нормальный инвойс.",
  "Я работаю по инвойсу от компании, а не по кошелькам."
];

const BARGAIN = [
  "Если дам сразу 10 кандидатов — будет скидка?",
  "При объёме в пятёрку людей двигаем цену?",
  "По цене давайте приземлимся. За одного столько не дам."
];

const CANDIDATE_TEST = [
  "Один тестовый кандидат — возможно, но сначала документы.",
  "Тест обсудим после Demand и контракта.",
  "Один на пробу — ок, но без бумаг не двигаюсь."
];

const CLOSING_READY = [
  "Хорошо, финализируем.",
  "Ок, шлите инвойс — проверю и двинемся.",
  "Готово. Жду реквизиты и счёт."
];

// МЯГКИЕ РЕПЛИКИ ДЛЯ GREETING (по доверию)
const GREETING_SOFT = [
  "Виктор, рад знакомству. Спасибо за визитку — гляну. Чем вы обычно помогаете, какие направления сильнее всего?",
  "Приятно познакомиться. Сайт открыт, посмотрю. Коротко: с какими работодателями вы чаще работаете?",
  "Спасибо, принял визитку. Расскажите, какие у вас сейчас основные вакансии и условия?"
];

const GREETING_INQUIRY = [
  "Вижу сайт. Чтобы понимать масштаб: с какими регионами/отраслями вы работаете чаще всего?",
  "Ок, принял. По вакансиям: это в основном CZ, PL? По зарплатам — какой коридор предлагаете?",
  "Супер. Тогда сориентируйте: какие документы обычно готовите на старте, кто подписывает?"
];

const GREETING_PRO = [
  "Посмотрел визитку. Давайте предметно: какие вакансии готовы закрыть первыми и на каких условиях?",
  "Окей. Какие документы готовы предоставить в первую очередь и кто работодатель?",
  "Договорились. Кого можем протестировать первым кандидатом и что по ставке?"
];

// Запретные паттерны
const BANNED_PATTERNS = [
  /кошел(е|ё)к|wallet/i,
  /переведите.*мне/i,
  /я оплачу первым/i,
  /гарантирую.*виз/i,
  /связи.*посольств/i
];

// ---------- Сборка сообщений в LLM ----------
function buildMessages({ history = [], message, trust, evidences }) {
  const sys = {
    role: 'system',
    content:
      SYSTEM_PROMPT +
      `\n\n[Контекст]\ntrust=${trust}; evidences=${JSON.stringify(evidences || [])}\n${PRICEBOOK}\n` +
      `Отвечай СТРОГО одним JSON-объектом (см. формат).`
  };

  const trimmed = history.slice(-12).map(h => ({
    role: h.role, content: h.content
  }));

  return [sys, ...trimmed, { role: 'user', content: message }];
}

// ---------- Обёртка с ретраями и таймаутом ----------
async function createChatWithRetry(payload, tries = 2) {
  let lastErr;
  while (tries--) {
    try {
      // sdk не всегда принимает второй аргумент опций — оставим один объект
      return await groq.chat.completions.create(payload);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr;
}

// ---------- «Оживление» и бортики ----------
function humanize({ parsed, trust, evidences, history }) {
  let reply = String(parsed.reply || '').trim();
  const lastBot = lastAssistantReplyFromHistory(history);
  const uniqEvidenceCount = new Set(evidences || []).size;
  const gatesOk = (trust >= 90 && uniqEvidenceCount >= 2);

  // Greeting: если только лёгкие пруфы (визитка/сайт) — мягко общаемся, не требуем контракт сразу
  const onlyLightProofs = uniqEvidenceCount >= 1 && !(evidences || []).some(e =>
    ['demand_letter','contract_pdf'].includes(String(e).toLowerCase())
  );

  const userText = (history || []).filter(h=>h.role==='user').slice(-1)[0]?.content || '';

  if ((parsed.stage === 'Greeting' || !parsed.stage) && onlyLightProofs) {
    if (trust < 40)       reply = pick(GREETING_SOFT);
    else if (trust < 70)  reply = pick(GREETING_INQUIRY);
    else                  reply = pick(GREETING_PRO);

    parsed.stage = 'Greeting';
    parsed.needEvidence = false;
    parsed.suggestedActions = ["ask_demands","ask_contract","ask_price_breakdown"];
    parsed.confidence = Math.max(parsed.confidence ?? 0, Math.min(70, trust));
  }

  // Запрещённые фразы → перехват в документы
  if (!reply || BANNED_PATTERNS.some(rx => rx.test(reply))) {
    reply = pick(ASK_DOCS_HUMAN);
    parsed.stage = parsed.stage === 'Payment' ? 'Contract' : (parsed.stage || 'Demand');
    parsed.needEvidence = true;
    parsed.suggestedActions = ["ask_demands","ask_contract"];
    parsed.confidence = Math.min(parsed.confidence ?? trust, 70);
  }

  // Ранняя оплата до «ворот»
  if (!gatesOk && parsed.stage === 'Payment') {
    reply = pick(TOO_EARLY_PAY_HUMAN);
    parsed.stage = 'Contract';
    parsed.needEvidence = true;
    const set = new Set(parsed.suggestedActions || []);
    set.add('ask_demands'); set.add('ask_contract'); set.add('ask_price_breakdown');
    parsed.suggestedActions = Array.from(set);
    parsed.confidence = Math.min(parsed.confidence ?? trust, 80);
  }

  // Подталкивания по последнему тексту пользователя
  if (/крипт|crypto|usdt|btc/i.test(userText)) reply ||= pick(CRYPTO_SKEPTIC);
  if (/сколько|цена|дорог|ценник|стоим/i.test(userText)) reply ||= pick(BARGAIN);
  if (/кандидат|people|workers|людей|тест/i.test(userText)) reply ||= pick(CANDIDATE_TEST);

  // Ворота пройдены → «финализируем»
  if (gatesOk && parsed.stage !== 'Payment') {
    const set = new Set(parsed.suggestedActions || []);
    set.add('invoice_request');
    parsed.suggestedActions = Array.from(set);
    reply ||= pick(CLOSING_READY);
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
    parsed.confidence = Math.max(parsed.confidence ?? 0, trust);
  }

  // Анти-повтор
  if (reply && lastBot && reply.trim().toLowerCase() === lastBot.trim().toLowerCase()) {
    reply = pick(FALLBACK_HUMAN);
  }

  // Финальный штрих
  parsed.reply = trimToSentences(reply, 6) || pick(FALLBACK_HUMAN);
  parsed.suggestedActions = normalizeActions(parsed.suggestedActions);

  // Если просим документы — stage не ниже Demand
  if (!gatesOk && /(Demand|контракт)/i.test(parsed.reply) && parsed.stage === 'Greeting') {
    parsed.stage = 'Demand';
  }

  return parsed;
}

// ---------- Вызов LLM ----------
async function runLLM({ history, message, evidences, stage }) {
  // ПОЛНАЯ история (с текстами), чтобы trust учитывал тон
  const trust = computeTrust({
    baseTrust: 20,
    evidences: evidences || [],
    history: history || [],
    lastUserText: message || ''
  });

  const messages = buildMessages({ history, message, trust, evidences });

  const resp = await createChatWithRetry({
    model: MODEL,
    temperature: 0.35,
    top_p: 0.9,
    frequency_penalty: 0.3,
    presence_penalty: 0.0,
    max_tokens: 380,
    response_format: { type: 'json_object' },
    messages
  });

  const raw = resp?.choices?.[0]?.message?.content || '{}';
  const json = extractFirstJsonObject(raw);

  let parsed;
  try {
    parsed = LLMShape.parse(json);
  } catch {
    parsed = null;
  }

  if (!parsed) {
    parsed = {
      reply: pick(FALLBACK_HUMAN),
      confidence: clamp(trust, 0, 40),
      stage: stage || "Greeting",
      needEvidence: true,
      suggestedActions: ["ask_demands","ask_contract"]
    };
  }

  // Страховки типов
  parsed.reply = String(parsed.reply || '').trim();
  parsed.stage = String(parsed.stage || stage || "Greeting");
  parsed.confidence = clamp(Number(parsed.confidence ?? trust), 0, 100);
  parsed.needEvidence = Boolean(parsed.needEvidence);
  parsed.suggestedActions = Array.isArray(parsed.suggestedActions) ? parsed.suggestedActions : [];

  // Оживляем и дожимаем логику
  parsed = humanize({ parsed, trust, evidences, history });

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

// Нормализация входа (общая)
function sanitizeHistory(arr){
  return Array.isArray(arr) ? arr.slice(-50).map(h => ({
    role: (h.role === 'assistant' ? 'assistant' : 'user'),
    content: String(h.content || '').replace(/<[^>]+>/g, ''),
    stage: h.stage ? String(h.stage) : undefined
  })) : [];
}
function normalizeEvidenceKey(k){
  const map = { contract:'contract_pdf', demand:'demand_letter', card:'business_card' };
  return map[String(k || '')] || String(k || '');
}

// /api/reply — основной
app.post('/api/reply', async (req, res) => {
  try {
    const b = req.body || {};

    // Сообщение
    const rawMessage = String(b.user_text ?? b.message ?? '').trim();
    if (!rawMessage || rawMessage.length > 2000) {
      return res.status(400).json({ ok: false, error: 'Invalid message length' });
    }

    // Evidences
    const evidences = Array.isArray(b.evidences)
      ? [...new Set(b.evidences.map(normalizeEvidenceKey).filter(Boolean))]
      : (Number.isFinite(b.evidence)
          ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`)
          : []);

    // История
    const history = sanitizeHistory(b.history);

    // Схема
    const dataForLLM = {
      sessionId: String(b.sessionId || 'default'),
      message: rawMessage,
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

    // считаем trust с ПОЛНОЙ историей (тон)
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

    // согласованная подсказка по «воротам»
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
      stage: data.stage
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
