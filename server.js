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
  max: 40,             // 40 req/IP/мин
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

// ---------- Groq ----------
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
const TEMPERATURE = Number(process.env.TEMPERATURE ?? 0.35);
const REPLY_MAX_TOKENS = Number(process.env.REPLY_MAX_TOKENS ?? 380);

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
const maybe = (arr, p=0.5) => (Math.random() < p ? pick(arr) : '');
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
function isQuestion(s=''){
  const t = s.trim();
  return /[?]\s*$/.test(t) || /\b(как|когда|сколько|почему|зачем|где|кто|что|какие|какой)\b/i.test(t);
}
function joinParts(parts){
  return parts.filter(Boolean).join(' ')
    .replace(/\s([.!?,])/g,'$1')
    .replace(/\s+/g,' ').trim();
}

// --- actions normalize (согласовано с prompt.js) ---
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

// ---------- Human phrasebanks (×5 разнообразия) ----------

// Общая эмпатия/подтверждение
const EMPATHY_ACKS = [
  "Понимаю.", "Слышу вас.", "Ок, принято.", "Хорошо, понял.",
  "Я услышал.", "Принял к сведению.", "Отмечу это.", "Записал.",
  "Да, это понятно.", "Логично."
];

// Смягчители/оговорки
const HEDGE_PHRASES = [
  "если честно", "по опыту", "как правило", "с моей стороны",
  "буду откровенен", "честно говоря", "по правде сказать",
  "на мой взгляд", "из практики"
];

// Ремонт коммуникации
const REPAIR_PHRASES = [
  "Кажется, я не так понял.", "Давайте уточню.",
  "Поправьте меня, если неверно понял.", "Сформулирую точнее.",
  "Чтобы не гадать — уточню."
];

// Small-talk (лёгкий, без воды)
const SMALLTALK_OPENERS = [
  "Кстати, сайт открылся нормально — без капчи, это уже плюс.",
  "Нью-Дели жарит, но работаем.",
  "Сегодня завал по письмам, но вашу визитку посмотрю.",
  "Сразу скажу: ценю конкретику и проверяемость.",
  "Если коротко: люблю документы больше, чем обещания."
];

// Лёгкий юмор (редко)
const LIGHT_HUMOR = [
  "Визитка симпатичная — не как те PDF без печати, что мне шлют.",
  "Комик Санс не заметил — это хорошо.",
  "Если буквы читаемы, уже приятнее, чем обычно :)",
  "PDF с печатью — как музыка для глаз."
];

// Благодарность за визитку
const THANK_FOR_CARD = [
  "Спасибо за визитку, посмотрю.",
  "Принял визитку, гляну детали.",
  "Окей, визитку получил.",
  "Визитку вижу, открыл.",
  "Принял. Отмечу у себя."
];

// Самопрезентация (коротко, 1 раз)
const SELF_BRIEF = [
  "Я Али, давно в визах, люблю конкретику и проверяемость.",
  "Я осторожный: видел разное, поэтому прошу документы.",
  "Коротко: без бумажек не бегу, но общаюсь нормально.",
  "Я не спешу, но и не тяну — если всё прозрачно."
];

// Микро-уточнения/зеркало (вопросы-«поддержки»)
const MICRO_QUESTIONS = [
  "Верно понимаю, речь про CZ/PL?",
  "Про работодателя расскажете чуть подробнее?",
  "Это ваш основной рынок сейчас?",
  "Ставки по вакансии — примерно какой коридор?",
  "Кто обычно подписывает со стороны работодателя?"
];

// Greeting-пулы по доверию (расширено)
const GREETING_SOFT = [
  "Расскажите коротко, с какими работодателями работаете?",
  "Какие у вас сейчас основные вакансии и условия?",
  "Понять бы масштаб: в каких отраслях вы сильнее?",
  "С чего начнём: вакансии или условия?"
];
const GREETING_WARM = [
  "Чтобы сориентироваться: CZ и PL — основные направления?",
  "По зарплатам: какой коридор обычно предлагаете?",
  "Какие документы вы готовите на старте обычно?",
  "Кто контакт на стороне работодателя?"
];
const GREETING_INQUIRY = [
  "Давайте предметно: какие вакансии готовы закрыть первыми?",
  "Какие документы готовы предоставить в первую очередь и кто работодатель?",
  "Кого можно протестировать первым кандидатом и на каких ставках?",
  "По срокам: что у вас в реальности получается?"
];

// Анти-крипта/торг/тест-кандидат/готовность финализации
const CRYPTO_SKEPTIC = [
  "Крипта — риск. Предпочитаю счёт-фактуру и банковский перевод.",
  "Крипта? Не лучший знак. Давайте нормальный инвойс.",
  "Я работаю по инвойсу от компании, а не по кошелькам."
];
const BARGAIN = [
  "Если дам сразу 10 кандидатов — будет скидка?",
  "При объёме в пятёрку людей двигаем цену?",
  "По цене давайте приземлимся. За одного столько не дам.",
  "А при повторных кейсах условия лучше?"
];
const CANDIDATE_TEST = [
  "Один тестовый кандидат — возможно, но сначала документы.",
  "Тест обсудим после Demand и контракта.",
  "Один на пробу — ок, но без бумаг не двигаюсь.",
  "Покажите документы — тогда один тест попробуем."
];
const CLOSING_READY = [
  "Хорошо, финализируем.",
  "Ок, шлите инвойс — проверю и двинемся.",
  "Готово. Жду реквизиты и счёт.",
  "Договорились. Счёт посмотрю, если всё ок — двинемся."
];

// Документы рано/запретные паттерны/ранняя оплата
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
const BANNED_PATTERNS = [
  /кошел(е|ё)к|wallet/i,
  /переведите.*мне/i,
  /я оплачу первым/i,
  /гарантирую.*виз/i,
  /связи.*посольств/i
];

// ---------- Session micro-memory ----------
const sessionState = new Map();
function sessionKey(history, fallback='default'){
  // попытка вытащить sid:XXXX из первой реплики; иначе fallback
  const sidLine = (history?.[0]?.content || '').match(/sid:([a-z0-9-_]+)/i);
  return sidLine?.[1] || fallback;
}
function getState(key){
  if (!sessionState.has(key)) {
    sessionState.set(key, {
      smalltalkUsed: false,
      humorQuota: 1,
      greetedOnce: false,
      mood: pick(['neutral','warm','wary','dry']), // эмоциональный модификатор
      lastTopic: '',
      lastActions: []
    });
  }
  return sessionState.get(key);
}

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

// ---------- Humanize (человечность ×5) ----------
function humanize({ parsed, trust, evidences, history }) {
  const lastUser = (history || []).filter(h=>h.role==='user').slice(-1)[0]?.content || '';
  const lastBot  = lastAssistantReplyFromHistory(history);
  const uniqEvidenceCount = new Set(evidences || []).size;
  const hasHard = (evidences||[]).some(e => /^(demand_letter|contract_pdf)$/i.test(String(e)));
  const onlyLightProofs = uniqEvidenceCount >= 1 && !hasHard;
  const gatesOk = (trust >= 90 && uniqEvidenceCount >= 2);

  // Память/настрой
  const sid = sessionKey(history, 'default');
  const S = getState(sid);

  // Базовый ответ LLM
  let reply = String(parsed.reply || '').trim();

  // Эмо-модулятор: чуть варьируем начало/темп (только не при финализации)
  const moodLead = {
    warm: maybe(EMPATHY_ACKS, 0.7),
    wary: maybe(HEDGE_PHRASES, 0.7),
    dry:  "",
    neutral: maybe(EMPATHY_ACKS, 0.4)
  }[S.mood] || '';

  // --- GREETING: мягко, живо, без требований ---
  if ((parsed.stage === 'Greeting' || !parsed.stage) && onlyLightProofs) {
    const parts = [];

    parts.push(pick(THANK_FOR_CARD));

    if (!S.greetedOnce) {
      parts.push(maybe(SELF_BRIEF, 0.95));
      S.greetedOnce = true;
    }

    if (!S.smalltalkUsed && trust >= 45) {
      parts.push(maybe(SMALLTALK_OPENERS, 0.6));
      S.smalltalkUsed = true;
    }

    // микро-зеркало + уточнение
    parts.push(maybe(MICRO_QUESTIONS, 0.6));

    // вопрос по делу с градацией доверия
    if (trust < 40)       parts.push(pick(GREETING_SOFT));
    else if (trust < 70)  parts.push(pick(GREETING_WARM));
    else                  parts.push(pick(GREETING_INQUIRY));

    reply = joinParts([moodLead, ...parts]);
    parsed.stage = 'Greeting';
    parsed.needEvidence = false;
    parsed.suggestedActions = ["ask_demands","ask_contract","ask_price_breakdown"];
    parsed.confidence = Math.max(parsed.confidence ?? 0, Math.min(75, trust));
  }

  // Мягкие оговорки/эмпатия почти всегда, пока не финализируем
  if (!gatesOk) {
    reply = joinParts([maybe(EMPATHY_ACKS, 0.5), maybe(HEDGE_PHRASES, 0.5), reply]);
  }

  // Редкий лёгкий юмор при хорошем доверии (без красных флагов)
  if (S.humorQuota > 0 && trust >= 72 && !/крипт|кошел|wallet|срочно платите|100%/i.test(lastUser)) {
    if (Math.random() < 0.28) { reply = joinParts([reply, maybe(LIGHT_HUMOR, 0.9)]); S.humorQuota--; }
  }

  // Запрещённые паттерны → мягкий перехват в документы
  if (!reply || BANNED_PATTERNS.some(rx => rx.test(reply))) {
    reply = pick(ASK_DOCS_HUMAN);
    parsed.stage = parsed.stage === 'Payment' ? 'Contract' : (parsed.stage || 'Demand');
    parsed.needEvidence = true;
    parsed.suggestedActions = ["ask_demands","ask_contract"];
    parsed.confidence = Math.min(parsed.confidence ?? trust, 70);
  }

  // Ранняя оплата до «ворот» → отбой
  if (!gatesOk && parsed.stage === 'Payment') {
    reply = pick(TOO_EARLY_PAY_HUMAN);
    parsed.stage = 'Contract';
    parsed.needEvidence = true;
    const set = new Set(parsed.suggestedActions || []);
    set.add('ask_demands'); set.add('ask_contract'); set.add('ask_price_breakdown');
    parsed.suggestedActions = Array.from(set);
    parsed.confidence = Math.min(parsed.confidence ?? trust, 80);
  }

  // Тематические подталкивания по последнему юзер-тексту
  if (/крипт|crypto|usdt|btc/i.test(lastUser)) reply ||= pick(CRYPTO_SKEPTIC);
  if (/сколько|цена|дорог|ценник|стоим/i.test(lastUser)) reply ||= pick(BARGAIN);
  if (/кандидат|people|workers|людей|тест/i.test(lastUser)) reply ||= pick(CANDIDATE_TEST);

  // Ворота пройдены → финализация
  if (gatesOk && parsed.stage !== 'Payment') {
    const set = new Set(parsed.suggestedActions || []);
    set.add('invoice_request');
    parsed.suggestedActions = Array.from(set);
    reply ||= pick(CLOSING_READY);
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
    parsed.confidence = Math.max(parsed.confidence ?? 0, trust);
  }

  // Ремонт коммуникации: если коротко/повтор
  const tooGeneric = reply.length < 24 || /^не понял|сформулируйте проще|повторите/i.test(reply);
  const sameAsBefore = (reply && lastBot && reply.trim().toLowerCase() === lastBot.trim().toLowerCase());
  if (tooGeneric || sameAsBefore) {
    reply = joinParts([pick(REPAIR_PHRASES), pick(GREETING_WARM)]);
  }

  // Анти-луп по действиям
  const actionsNow = normalizeActions(parsed.suggestedActions);
  if (JSON.stringify(actionsNow) === JSON.stringify(S.lastActions)) {
    // заменим формулировку, чтобы не звучало одинаково
    reply = joinParts([maybe(EMPATHY_ACKS, 0.6), reply]);
  }
  S.lastActions = actionsNow;

  // Финальный штрих
  parsed.reply = trimToSentences(reply, 6) || pick(REPAIR_PHRASES);
  parsed.suggestedActions = actionsNow;

  // Если просим документы — stage не ниже Demand
  if (!gatesOk && /(Demand|контракт)/i.test(parsed.reply) && parsed.stage === 'Greeting') {
    parsed.stage = 'Demand';
  }

  return parsed;
}

// ---------- Вызов LLM ----------
async function runLLM({ history, message, evidences, stage, sessionId='default' }) {
  // Полная история для тонального trust
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
    frequency_penalty: 0.3,
    presence_penalty: 0.0,
    max_tokens: REPLY_MAX_TOKENS,
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

  // Fail-soft по стадии
  if (!parsed) {
    const fallbackByStage = {
      Greeting: pick(GREETING_SOFT),
      Demand: pick(ASK_DOCS_HUMAN),
      Contract: pick(ASK_DOCS_HUMAN),
      Payment: pick(TOO_EARLY_PAY_HUMAN)
    };
    const fb = fallbackByStage[stage || 'Greeting'] || "Не понял. Давайте конкретнее.";
    parsed = {
      reply: fb,
      confidence: clamp(trust, 0, 45),
      stage: stage || "Greeting",
      needEvidence: stage === 'Payment' ? true : false,
      suggestedActions: ["ask_demands","ask_contract"]
    };
  }

  // Страховки типов
  parsed.reply = String(parsed.reply || '').trim();
  parsed.stage = String(parsed.stage || stage || "Greeting");
  parsed.confidence = clamp(Number(parsed.confidence ?? trust), 0, 100);
  parsed.needEvidence = Boolean(parsed.needEvidence);
  parsed.suggestedActions = Array.isArray(parsed.suggestedActions) ? parsed.suggestedActions : [];

  // Оживляем
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

// Helpers: нормализация
function sanitizeHistory(arr){
  return Array.isArray(arr) ? arr.slice(-50).map(h => ({
    role: (h.role === 'assistant' ? 'assistant' : 'user'),
    content: String(h.content || '').replace(/<[^>]+>/g, ''),
    stage: h.stage ? String(h.stage) : undefined
  })) : [];
}
function normalizeEvidenceKey(k){
  const map = {
    contract:'contract_pdf', contractpdf:'contract_pdf', договор:'contract_pdf',
    demand:'demand_letter', demandletter:'demand_letter', деманд:'demand_letter',
    card:'business_card', визитка:'business_card', сайт:'website'
  };
  const key = String(k || '').toLowerCase().trim();
  return map[key] || key;
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
      stage: parsed.stage,
      sessionId: parsed.sessionId
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
