// server.js — RenovoGo LLM backend (stable memory, no robotic ACKs, realistic flow)
// v2025-09-16-7

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

// ---------- PRICEBOOK (fees, not salaries) ----------
const PRICEBOOK = `
[PRICEBOOK v1 — CZ/PL (fees, not salaries)]
— Czech Republic (service fees per candidate):
  • 3m €270 + €150  • 6m €300 + €150  • 9m €350 + €150
  • 24m €350 + €350
  • Embassy reg (LT only): €500 = €250 + €250 (refund €250 if >6m no slot)
— Poland (service fees):
  • 9m seasonal €350 + €150  • 12m €350 + €350
— General: free verification; every PDF has verify guidelines; all under CZ/EU law.
— NOTE: Service fees are NOT employee salary. Never mix fees with wages.
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

function logError(err, ctx=''){
  console.error(`[${new Date().toISOString()}] ${ctx}:`, err?.stack || err);
}

function forceMasculine(text){
  return String(text||'')
    .replace(/\bрада\b/gi, 'рад')
    .replace(/\bготова\b/gi, 'готов')
    .replace(/\bсмогла\b/gi, 'смог')
    .replace(/\bмогла\b/gi, 'мог');
}
function splitSentences(t=''){
  return String(t).split(/(?<=[.!?])\s+/).filter(s => s.trim());
}
function joinUniqueSentences(chunks=[]){
  const seen = new Set(); const out = [];
  for (const s of chunks.flatMap(splitSentences)) {
    const k = s.trim().toLowerCase();
    if (k && !seen.has(k)) { seen.add(k); out.push(s.trim()); }
  }
  return out.join(' ');
}
function stripRequisitesFromDemand(t=''){
  // убираем любые фразы «реквизиты из Demand»
  return String(t)
    .replace(/[^.?!]*реквизит[^.?!]*(из\s+)?demand[^.?!]*[.?!]/gi, '')
    .replace(/\s{2,}/g,' ')
    .trim();
}
// убираем запросы «реквизиты работодателя/емployer requisites»
function stripEmployerRequisitesRequests(t=''){
  return String(t).replace(
    /[^.?!]*(реквизит\w*|requisite\w*)[^.?!]*(работодател\w*|employer)[^.?!]*[.?!]/gi,
    ''
  );
}
// убираем «роботские» ACK’и типа «спасибо/получил визитку/деманд/контракт/видео»
function stripRoboticAcks(t=''){
  const KEY = '(demand|деманд|business\\s*card|визитк|контракт|соглашен|sample|презентац|video|видео|виза|pdf)';
  const r1 = new RegExp(`[^.?!]*\\b(спасибо|получил|получена|получено|принял|received|got)\\b[^.?!]*${KEY}[^.?!]*[.?!]`,'gi');
  const r2 = new RegExp(`[^.?!]*${KEY}[^.?!]*\\b(получил|получена|получено|принял|received|got)\\b[^.?!]*[.?!]`,'gi');
  return String(t).replace(r1,'').replace(r2,'').replace(/\s{2,}/g,' ').trim();
}
function cleanSales(t=''){
  return String(t).replace(/(?:оставьте заявку.*?|мы предлагаем широкий спектр услуг)/gi, '').trim();
}
function conciseJoin(parts){
  return parts.filter(Boolean).map(s=>String(s).trim()).filter(Boolean).join(' ');
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
  return `По долгосрочному — ${REG_LONGTERM_MONTHS} мес назад; по сезонному — ${REG_SEASONAL_MONTHS} мес назад. Очереди нестабильные. Сначала проверю Demand Letter и контракт.`;
}

// ---------- Микро-состояние сессии ----------
/*
  sessionState: Map<sid, {
    lastReply: string,
    lastActions: string[],
    seenEvidences: Map<key, { count: number, lastAt: number }>,
    evidenceDetails: Record<string, any>
  }>
*/
const sessionState = new Map();
function getState(sid='default'){
  if (!sessionState.has(sid)) {
    sessionState.set(sid, {
      lastReply: '',
      lastActions: [],
      seenEvidences: new Map(),
      evidenceDetails: Object.create(null)
    });
  }
  return sessionState.get(sid);
}

function bumpEvidence(sid, key, details){
  const S = getState(sid);
  const rec = S.seenEvidences.get(key) || { count: 0, lastAt: 0 };
  rec.count += 1;
  rec.lastAt = Date.now();
  S.seenEvidences.set(key, rec);
  if (details && typeof details === 'object') {
    S.evidenceDetails[key] = { ...(S.evidenceDetails[key]||{}), ...details };
  }
  return rec.count;
}
function evidenceCountUnique(sid){
  return getState(sid).seenEvidences.size;
}
function hasEvidence(sid, key){
  return getState(sid).seenEvidences.has(key);
}

// ---------- Сборка сообщений в LLM ----------
function buildMessages({ history = [], message, trust, evidences, evidenceDetails }) {
  const sys = {
    role: 'system',
    content:
      SYSTEM_PROMPT +
      `\n\n[Контекст]\n` +
      `— trust=${trust}; evidences=${JSON.stringify(evidences || [])}\n` +
      `— evidence_details=${JSON.stringify(evidenceDetails || {})}\n` +
      PRICEBOOK + `\n` +
      `Правила: Demand Letter — это запрос/описание вакансии (без реквизитов работодателя). ` +
      `Реквизиты и стоимость — только в контракте о сотрудничестве/инвойсе. ` +
      `Отвечай СТРОГО одним JSON-объектом. Будь кратким (до ${MAX_SENTENCES} предложений).`
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

// ---------- Пост-правила ----------
function postRules({ parsed, trust, evidences, history, userText, sid, evidenceDetails }) {
  const S = getState(sid);

  // === init state / scores / counters ===
  S.meta   = S.meta   || {};
  S.scores = S.scores || {};
  if (typeof S.scores.trust !== 'number')   S.scores.trust = 50;
  if (typeof S.scores.rapport !== 'number') S.scores.rapport = 60;
  S.meta.totalUserMsgs = (S.meta.totalUserMsgs || 0) + 1;
  S.seenEvidences = S.seenEvidences || new Set();

  // Новые пруфы ТОЛЬКО в этом запросе
  const inc = new Set((evidences || []).filter(k => !S.seenEvidences.has(k)));

  let reply = String(parsed.reply || '').trim();
  const setActions = new Set(parsed.suggestedActions || []);

  // Быстрый ответ «кто ты?»
  if (/(как.*зовут|вас зовут|ваше имя|who are you)/i.test(userText)) {
    reply = 'Меня зовут Али.';
    parsed.stage ??= 'Greeting';
  }

  // Регистрации/слоты — консистентный ответ
  if (/(когда|последн).*(регистрир|записыва)|слот|очеред/i.test(userText)) {
    reply = registrationAnswer();
    parsed.stage = 'Demand';
    parsed.needEvidence = true;
    parsed.suggestedActions = ['ask_demands','ask_coop_contract'];
  }

  // Вопрос про кол-во кандидатов
  const askCandidates = /(сколько.*кандидат|кандидат(ов)?\s*(есть|готов|подад)|сколько человек будем подавать)/i.test(userText);
  if (askCandidates) {
    if (hasEvidence(sid,'demand_letter') && hasEvidence(sid,'coop_contract_pdf')) {
      reply = 'Готов подать 1 кандидата на старт в Чехию. Остальное после того как проверю вашу работу.';
      parsed.stage = parsed.stage === 'Payment' ? 'Payment' : 'Candidate';
      parsed.needEvidence = false;
      setActions.add('invoice_request');
    } else {
      const miss = [];
      if (!hasEvidence(sid,'demand_letter'))    miss.push('Demand Letter');
      if (!hasEvidence(sid,'coop_contract_pdf')) miss.push('полный контракт о сотрудничестве');
      reply = `Подтвержу точное число сразу после ${miss.join(' и ')}. Сейчас ориентир — 1–2 кандидата на старт.`;
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      parsed.suggestedActions = ['ask_demands','ask_coop_contract'];
    }
  }

  // ==== Hesitation lines (когда он не готов платить) ====
  const HESITATION_LINES = [
    'Понимаю, но для меня это пока дорого.',
    'Пожалуйста, не давите. Если решу работать с вами — обязательно сообщу.',
    'У меня пока нет кандидатов.',
    'Сейчас занимаюсь подбором персонала под ваши запросы — напишу на следующей неделе.',
    'Для меня это дорого, могу заплатить после визы.',
    'Готов оплатить после разрешений на работу. Сейчас рисковать такой суммой не хочу.',
    'Пока не уверен, что хочу работать именно с вами.',
    'Я отпрвил Ваш запрос пратнерам, пока нет кандидатов.'
  ];

  // Псевдослучайный выбор без повторов подряд + анти-спам
  function pickHesitation(state) {
    if (!state.meta) state.meta = {};
    const last = state.meta.lastHesitationLine;
    const now  = Date.now();
    const lastTs       = state.meta.lastHesitationTs || 0;
    const lastMsgCount = state.meta.lastHesitationMsgCount || 0;
    const canUseByTime = (now - lastTs) > 120000; // 2 мин
    const canUseByMsgs = (state.meta.totalUserMsgs || 0) - lastMsgCount >= 4;
    const canUse = canUseByTime || canUseByMsgs;

    if (!canUse) {
      return { line: 'К оплате вернёмся после проверки документов и согласования контракта.', counted: false };
    }

    const pool = last ? HESITATION_LINES.filter(l => l !== last) : [...HESITATION_LINES];
    const line = pool[Math.floor(Math.random() * pool.length)];

    state.meta.lastHesitationLine = line;
    state.meta.lastHesitationTs = now;
    state.meta.lastHesitationMsgCount = (state.meta.totalUserMsgs || 0);

    return { line, counted: true };
  }

  // === Guard: не инициировать оплату при низком доверии ===
  {
    const currentTrust = Number.isFinite(trust) ? trust : (S.scores?.trust ?? 50);
    if (parsed.stage === 'Payment' && currentTrust < 90) {
      const picked = pickHesitation(S);
      reply = picked.line;
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      setActions.add('ask_demands');
      setActions.add('ask_coop_contract');
      setActions.add('ask_price_breakdown');
      if (picked.counted) {
        S.scores.rapport = Math.max(0, (S.scores.rapport || 0) - 10); // штраф только к раппорту
      }
    }
  }

  // ====== Тихая фиксация приходящих материалов (без «получил/спасибо») ======
  // Визитка — тихо сохраняем
  if (inc.has('business_card') || (evidenceDetails && evidenceDetails.business_card)) {
    bumpEvidence(sid, 'business_card', evidenceDetails?.business_card);
  }

  // Demand Letter — тихо учитываем. Если ещё нет coop-контракта — мягко направляем на него
  if (inc.has('demand_letter')) {
    bumpEvidence(sid, 'demand_letter');
    if (!hasEvidence(sid,'coop_contract_pdf')) {
      setActions.add('ask_coop_contract');
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      reply = joinUniqueSentences([reply, 'Пожалуйста, отправьте оффер или контракт с условиями сотрудничества.']);
    }
  }

  // Пример контракта — тихо учитываем; просим полный контракт
  if (inc.has('sample_contract_pdf')) {
    bumpEvidence(sid, 'sample_contract_pdf');
    if (!hasEvidence(sid,'coop_contract_pdf')) {
      setActions.add('ask_coop_contract');
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      reply = joinUniqueSentences([reply, 'Могли бы вы отправить контракт B2B?']);
    }
  }

  // Полный контракт — тихо учитываем; направляем к цене/инвойсу
  if (inc.has('coop_contract_pdf')) {
    bumpEvidence(sid, 'coop_contract_pdf');
    parsed.stage = 'Contract';
    parsed.needEvidence = false;
    setActions.add('ask_price_breakdown');
    reply = joinUniqueSentences([reply, 'Спасибо, ознакомился с условиями контракта. Пока не уверен, стоит работать или нет.']);
  }

  // Прочие материалы — тихо учитываем
  for (const key of ['visa_sample','presentation','video','website','company_registry','reviews','registry_proof','price_breakdown','slot_plan','invoice_template','nda']) {
    if (inc.has(key)) bumpEvidence(sid, key, evidenceDetails?.[key]);
  }

  // ====== Политика оплаты — только банк ======
  if (/(банк|банковск|crypto|крипто|usdt|btc|eth|криптовалют)/i.test(userText) && /оплат|плат[её]ж|инвойс|сч[её]т/i.test(userText)) {
    reply = 'Банковский инвойс. Криптовалюту не принимаем.';
    setActions.add('invoice_request');
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  // ====== Санитария текста (убираем лишнее и «робота») ======
  reply = stripEmployerRequisitesRequests(reply); // не просим «реквизиты работодателя»
  reply = stripRequisitesFromDemand(reply);       // не просим «реквизиты из Demand»
  reply = stripRoboticAcks(reply);                // убираем «спасибо/получил X» на каждое сообщение
  reply = cleanSales(reply);
  reply = forceMasculine(reply);
  reply = limitSentences(reply, MAX_SENTENCES);

  // Анти-луп
  if (reply && S.lastReply && reply.toLowerCase() === S.lastReply.toLowerCase()) {
    reply = 'Коротко: документы учёл. Готов подать 2 кандидата и перейти к оплате.';
    parsed.stage = 'Payment';
    setActions.add('invoice_request');
  }
  S.lastReply = reply;

  // Финальные ворота — контракт + ≥2 уникальных пруфа + trust≥90
  const uniqEvidenceCount = evidenceCountUnique(sid);
  const hasCoop = hasEvidence(sid,'coop_contract_pdf');
  const gatesOk = ( (Number.isFinite(trust) ? trust : S.scores.trust) >= 90 && uniqEvidenceCount >= 2 && hasCoop );
  if (gatesOk) {
    setActions.add('invoice_request');
    if (!/инвойс|сч[её]т|реквизит/i.test(reply)) {
      reply = conciseJoin([reply, 'Пришлите реквизиты для оплаты.']);
    }
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  parsed.reply = reply.trim();
  parsed.suggestedActions = normalizeActions(Array.from(new Set([...(parsed.suggestedActions||[]), ...setActions])));

  // Если просим документы — stage не ниже Demand
  if (/(demand|контракт|документ|полный контракт|сотрудничеств)/i.test(parsed.reply) && (!parsed.stage || parsed.stage === 'Greeting')) {
    parsed.stage = 'Demand';
  }

  return { parsed, trust: S.scores.trust };
}

// ---------- Вызов LLM ----------
async function runLLM({ history, message, evidences, stage, sessionId='default', evidenceDetails }) {
  const trust = computeTrust({
    baseTrust: 20,
    evidences: Array.from(new Set(evidences || [])), // только уникальные ключи влияют
    history: history || [],
    lastUserText: message || ''
  });

  const messages = buildMessages({ history, message, trust, evidences, evidenceDetails });

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
  parsed = postRules({
    parsed,
    trust,
    evidences,
    history,
    userText: message || '',
    sid: sessionId || 'default',
    evidenceDetails
  });

  return { trust, evidenceCount: evidenceCountUnique(sessionId), result: parsed };
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
    // Business card / визитка менеджера
    ['card','business_card'], ['визитка','business_card'], ['business_card','business_card'],

    // Demand
    ['demand','demand_letter'], ['demandletter','demand_letter'], ['деманд','demand_letter'],

    // Contracts
    ['sample','sample_contract_pdf'], ['sample_contract','sample_contract_pdf'],
    ['contract_sample','sample_contract_pdf'], ['пример_контракта','sample_contract_pdf'],
    ['contract_pdf','coop_contract_pdf'], // алиас со старого фронта
    ['contract','coop_contract_pdf'], ['contractpdf','coop_contract_pdf'], ['договор','coop_contract_pdf'],
    ['coop_contract','coop_contract_pdf'], ['full_contract','coop_contract_pdf'],
    ['контракт_о_сотрудничестве','coop_contract_pdf'],

    // Other proofs
    ['visa','visa_sample'], ['visa_scan','visa_sample'], ['пример_визы','visa_sample'], ['visa_sample','visa_sample'],
    ['site','website'], ['сайт','website'], ['website','website'],
    ['reviews','reviews'], ['отзывы','reviews'],
    ['registry','registry_proof'], ['uradprace','registry_proof'], ['регистрация','registry_proof'],
    ['presentation','presentation'], ['deck','presentation'], ['video','video'], ['youtube','video'],
    ['price','price_breakdown'], ['slot_plan','slot_plan'], ['company_registry','company_registry'],
    ['invoice_template','invoice_template'], ['nda','nda']
  ]);
  return map.get(key) || key;
}

// ---------- /api/reply ----------
/*
  Доп. поля:
  — evidence_details (object), например:
     { business_card: { name, phone, email, office, company }, website:{ url } }
  — evidence (число) — совместимость со старым фронтом (генерит proof_1..N)
*/
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

    const evidenceDetails = (b.evidence_details && typeof b.evidence_details === 'object') ? b.evidence_details : {};

    const history = sanitizeHistory(b.history);
    const sid = String(b.sessionId || 'default');

    const { trust, evidenceCount, result } = await runLLM({
      history,
      message: rawMessage,
      evidences,
      stage: b.stage,
      sessionId: sid,
      evidenceDetails
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

// ---------- /api/score ----------
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
    bad.push('Не смешивайте сервисные платежи (€270/€300/€350/€500) с зарплатой работника — это разные вещи.');

    res.json({ final, good, bad, trust, evidences: evidences.length });
  } catch (e) {
    logError(e, '/api/score');
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// ---------- Совместимость со старым роутом ----------
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
