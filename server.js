// server.js — RenovoGo LLM backend (stable memory, cold client Ali, anti-repeats)
// v2025-09-16-11

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 1. БАЗА: импорты, app, CORS, мини-рейт-лимит
   ────────────────────────────────────────────────────────────── */

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

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 2. МОДЕЛЬ, ПРАЙС, СХЕМЫ
   ────────────────────────────────────────────────────────────── */

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
const TEMPERATURE = Number(process.env.TEMPERATURE ?? 0.2);
const REPLY_MAX_TOKENS = Number(process.env.REPLY_MAX_TOKENS ?? 320);
const MAX_SENTENCES = Number(process.env.MAX_SENTENCES ?? 4);

// PRICEBOOK остаётся только для внутреннего контекста модели;
// Али как клиент НИКОГДА не озвучивает цены и не инициирует оплату.
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

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 3. УТИЛИТЫ, МИКРО-ПАМЯТЬ, ПАРСЕР DEMAND
   ────────────────────────────────────────────────────────────── */

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

// — анти-«женский род» (персона Али — мужчина)
function forceMasculine(text){
  return String(text||'')
    .replace(/\bрада\b/gi, 'рад')
    .replace(/\bготова\b/gi, 'готов')
    .replace(/\bсогласна\b/gi, 'согласен')
    .replace(/\bсмогла\b/gi, 'смог')
    .replace(/\bмогла\b/gi, 'мог')
    .replace(/\bприняла\b/gi, 'принял');
}

// [ALI-CLIENT] Нормализация и запрет «продажных» слов у клиента
function stripSalesy(text=''){
  let t = String(text);
  const salesy = [
    /(?:мы|у\s*нас)\s+предлагаем/i,
    /оставьте\s+заявку/i,
    /наш\s+пакет/i,
    /мы\s+сделаем/i,
    /мы\s+предоставим/i,
    /скидк/i,
    /акци/i
  ];
  for (const r of salesy) t = t.replace(r, '').trim();
  return t.replace(/\s{2,}/g, ' ');
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
  return String(t)
    .replace(/[^.?!]*реквизит[^.?!]*(из\s+)?demand[^.?!]*[.?!]/gi, '')
    .replace(/\s{2,}/g,' ')
    .trim();
}
function stripEmployerRequisitesRequests(t=''){
  return String(t).replace(
    /[^.?!]*(реквизит\w*|requisite\w*)[^.?!]*(работодател\w*|employer)[^.?!]*[.?!]/gi,
    ''
  );
}
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

// — сид-рандом по sessionId
function seededRand(str=''){
  let h = 2166136261 >>> 0;
  for (let i=0;i<str.length;i++){ h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return () => {
    h ^= h << 13; h ^= h >>> 17; h ^= h << 5;
    return ((h >>> 0) % 1000) / 1000;
  };
}

/* микропамять сессии */
const sessionState = new Map();
/*
  sessionState: Map<sid, {
    lastReply: string,
    lastActions: string[],
    seenEvidences: Map<key, { count: number, lastAt: number }>,
    evidenceDetails: Record<string, any>,
    lastObjection: string,
    demandFacts: Record<string, any>,
    turn: number,
    repeatStats: {
      phraseCounts: Map<string, number>,
      lastUsedTurn: Map<string, number>,
      topicCounts: Record<string, number>
    },
    alreadyCommitted: boolean
  }>
*/
function getState(sid='default'){
  if (!sessionState.has(sid)) {
    sessionState.set(sid, {
      lastReply: '',
      lastActions: [],
      seenEvidences: new Map(),
      evidenceDetails: Object.create(null),
      lastObjection: '',
      demandFacts: Object.create(null),
      turn: 0,
      repeatStats: {
        phraseCounts: new Map(),
        lastUsedTurn: new Map(),
        topicCounts: Object.create(null)
      },
      alreadyCommitted: false
    });
  }
  return sessionState.get(sid);
}

/* учёт «доказательств» */
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

/* DEMAND: хранение и парсинг */
function getDemandFacts(sid){ return getState(sid).demandFacts || {}; }
function setDemandFacts(sid, facts={}){
  const S = getState(sid);
  S.demandFacts = { ...(S.demandFacts||{}), ...facts };
  return S.demandFacts;
}
const num = s => {
  const m = String(s||'').replace(/\s+/g,'').replace(',', '.').match(/[\d.]+/);
  return m ? Number(m[0]) : undefined;
};
function extractDemandFactsFromDetails(details={}){
  const out = {};
  const meta = details?.demand_meta;
  const text = details?.demand_text || '';

  if (meta && typeof meta === 'object') {
    if (meta.position) out.position = String(meta.position).trim();
    if (meta.job_description) out.job_description = String(meta.job_description).trim();
    if (meta.salary_net_czk || meta.salary_net_eur) {
      out.salary = meta.salary_net_czk ? {value:num(meta.salary_net_czk), currency:'CZK'}
                                       : {value:num(meta.salary_net_eur), currency:'EUR'};
    }
    if (meta.accommodation_eur || meta.accommodation) {
      const v = meta.accommodation_eur ?? meta.accommodation;
      out.accommodation = { cost_eur: num(v) };
    }
    if (meta.transport_to_work) out.transport = String(meta.transport_to_work).trim();
    if (meta.period) out.period = String(meta.period).trim();
    if (meta.hours_monthly) out.hours_monthly = num(meta.hours_monthly);
    if (meta.schedule) out.schedule = String(meta.schedule).trim();
    if (meta.location) out.location = String(meta.location).trim();
  }

  const t = String(text);
  if (!out.position)   { const m = t.match(/Position[:\s-]*([^\n]+)/i); if (m) out.position = m[1].trim(); }
  if (!out.salary)     { const m = t.match(/Salary\s*(?:net)?[:\s-]*([^\n]+)/i); if (m){ const v = m[1]; out.salary = /czk/i.test(v) ? {value:num(v), currency:'CZK'} : {value:num(v), currency:'EUR'}; } }
  if (!out.accommodation){ const m = t.match(/Accommod(?:ation)?[:\s-]*([^\n]+)/i); if (m){ const v = m[1]; const eur = v.match(/(\d[\d\s.,]*)\s*(?:€|eur)/i); if (eur) out.accommodation = { cost_eur:num(eur[1]) }; } }
  if (!out.hours_monthly){ const m = t.match(/Working\s*hours\s*monthly[:\s-]*([^\n]+)/i); if (m) out.hours_monthly = num(m[1]); }
  if (!out.schedule)   { const m = t.match(/Workhours[:\s-]*([^\n]+)/i) || t.match(/Workday[:\s-]*([^\n]+)/i); if (m) out.schedule = m[1].trim(); }
  if (!out.period)     { const m = t.match(/Employment\s*Period[:\s-]*([^\n]+)/i); if (m) out.period = m[1].trim(); }
  if (!out.location)   { const m = t.match(/Location\s*of\s*work[:\s-]*([^\n]+)/i) || t.match(/Location[:\s-]*([^\n]+)/i); if (m) out.location = m[1].trim(); }

  return out;
}
function formatFactsShort(facts={}, topic='all'){
  const f = facts || {};
  const salaryStr = f.salary?.value ? `нетто от ${f.salary.value} ${f.salary.currency}` : null;
  const accomStr  = (f.accommodation?.cost_eur ? `жильё ~€${f.accommodation.cost_eur}/мес` : null);
  const hoursStr  = (f.hours_monthly ? `~${f.hours_monthly} ч/мес` : null);
  const schedStr  = (f.schedule ? `${f.schedule}` : null);
  const posStr    = (f.position ? `${f.position}` : null);
  const locStr    = (f.location ? `${f.location}` : null);
  const periodStr = (f.period ? `${f.period}` : null);

  if (topic === 'salary' && salaryStr) return `По деманду: ${salaryStr}.`;
  if (topic === 'accommodation' && accomStr) return `По жилью из деманда: ${accomStr}.`;
  if (topic === 'hours' && (hoursStr || schedStr)) return `График по деманду: ${[hoursStr, schedStr].filter(Boolean).join(', ')}.`;
  if (topic === 'location' && locStr) return `Локация в деманде: ${locStr}.`;

  const line1 = [posStr, locStr, periodStr].filter(Boolean).join(' • ');
  const line2 = [salaryStr, accomStr, hoursStr || schedStr].filter(Boolean).join(' • ');
  if (line1 && line2) return `По деманду вижу: ${line1}. Условия: ${line2}.`;
  if (line1) return `По деманду вижу: ${line1}.`;
  if (line2) return `Условия по деманду: ${line2}.`;
  return '';
}

/* Истории: «100% гарантия» — корректируем субъект на «агент/мошенник» */
function fixGuaranteeStoryText(t=''){
  return String(t)
    .replace(/клиент\s+(обещал|дал|гарантир\w*)/gi, 'агент обещал')
    .replace(/100%\s*гарант\w+/gi, '«гарантию» (что само по себе подозрительно)');
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 4. СТАДИИ, КОНСТАНТЫ, ПРИВЕТСТВИЕ
   ────────────────────────────────────────────────────────────── */

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

const REG_LONGTERM_MONTHS = 6;
const REG_SEASONAL_MONTHS = 3;

function registrationAnswer(){
  return `По долгосрочному — ${REG_LONGTERM_MONTHS} мес назад; по сезонному — ${REG_SEASONAL_MONTHS} мес назад. Очереди нестабильные.`;
}

// [ALI-CLIENT] Greeting — коротко и тепло, если собеседник представился (без «не спешить»)
function craftHumanGreeting({ base='', userText='' } = {}){
  const intro = /(меня зовут|я\s+(генераль|менедж|представител)|renovogo)/i.test(userText);
  if (intro) return 'Виктор, рад знакомству. Слушаю вас.';
  const variants = ['Здравствуйте.', 'Привет.', 'Добрый день.', 'Слушаю.'];
  const first = String(base||'').trim();
  return forceMasculine(splitSentences(first)[0] || variants[Math.floor(Math.random()*variants.length)]);
}

// [ALI-CLIENT] Больше не переписываем в «Какие вакансии…» — оставляем как есть
function rewriteVacancyQuestionToSupplierRole(text=''){
  return String(text||'').trim();
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 5. LLM: сборка сообщений, ретраи, пост-правила
   ────────────────────────────────────────────────────────────── */

// Сообщения для модели
function buildMessages({ history = [], message, trust, evidences, evidenceDetails }) {
  const sys = {
    role: 'system',
    content:
      SYSTEM_PROMPT +
      `\n\n[Контекст]\n` +
      `— trust=${trust}; evidences=${JSON.stringify(evidences || [])}\n` +
      `— evidence_details=${JSON.stringify(evidenceDetails || {})}\n` +
      PRICEBOOK + `\n` +
      // Greeting: без опросника/продажи, максимум 4 предложения, JSON only
      `Правила Greeting: короткое приветствие без требований и без продажи. ` +
      `Документы проси только реактивно (если собеседник сам спросил «что нужно»). ` +
      `Никогда не инициируй оплату и не озвучивай цены. ` +
      `Отвечай СТРОГО одним JSON-объектом. Будь кратким (до ${MAX_SENTENCES} предложений).`
  };
  const trimmed = (history||[]).slice(-12).map(h => ({ role: h.role, content: h.content }));
  return [sys, ...trimmed, { role: 'user', content: message }];
}

async function createChatWithRetry(payload, tries = 2) {
  let lastErr;
  while (tries--) {
    try { return await groq.chat.completions.create(payload); }
    catch (e) { lastErr = e; }
  }
  throw lastErr;
}

/** Жёсткий редиректор любых формулировок «контракта/реквизитов работодателя» → наш B2B контракт */
function redirectEmployerContractToCoop(text=''){
  let t = String(text || '');
  t = t.replace(
    /\b(контракт(?!\s*о\s*сотрудничестве)[^.!?\n]*\b(работодател[ьяею]|employer|company)[^.!?\n]*)/gi,
    'контракт о сотрудничестве (наш B2B)'
  );
  t = t.replace(
    /\b(реквизит\w*\s*(работодател[ьяею]|employer|company)|employer\s*(details|requisites))\b/gi,
    'реквизиты для инвойса (наши B2B)'
  );
  t = t.replace(
    /\b(документ\w*\s*(работодател[ьяею]|employer|company))\b/gi,
    'B2B-документы по сотрудничеству'
  );
  return t;
}

/* анти-повторы и кулдауны */
const STOP_PHRASES = [
  'как вы?',
  'ищете работу в польше',
  'ищете работу в чехии',
  'какие вакансии у вас доступны?',
  'какие вакансии у вас сейчас открыты?'
];
const MAX_SAME_PHRASE   = 1; // за всю сессию
const COOLDOWN_TURNS    = 6; // кулдаун повторной фразы
const MAX_QUESTIONS_IN_MSG = 1;

function normPhrase(s){ return String(s||'').toLowerCase().replace(/[^\p{L}\p{N}\s?.!,-]/gu,'').trim(); }

function repetitionGuard(reply, sid){
  const S = getState(sid);
  const { phraseCounts, lastUsedTurn } = S.repeatStats;
  const turn = S.turn;

  let sentences = splitSentences(reply);
  const out = [];
  let questionsUsed = 0;

  for (let s of sentences){
    const ns = normPhrase(s);

    if (STOP_PHRASES.some(p => ns.includes(p))) continue;

    const lastTurn = lastUsedTurn.get(ns) ?? -999;
    if (turn - lastTurn < COOLDOWN_TURNS) continue;

    const cnt = phraseCounts.get(ns) ?? 0;
    if (cnt >= MAX_SAME_PHRASE) continue;

    if (/\?\s*$/.test(s)) {
      if (questionsUsed >= MAX_QUESTIONS_IN_MSG) continue;
      questionsUsed++;
    }

    out.push(s);
    phraseCounts.set(ns, cnt + 1);
    lastUsedTurn.set(ns, turn);
  }

  if (out.length === 0) return 'Ок.';
  return out.join(' ');
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 5.1. POST-RULES (главная логика «клиента»)
   ────────────────────────────────────────────────────────────── */

function chooseObjection({ sid, userText='', trust=0, uniqEvidence=0, hasDemand=false, hasCoop=false, stage='Greeting' }){
  const S = getState(sid);
  const rnd = seededRand(sid);
  const hasPriceTalk   = /(цена|стоим|дорог|price|€|eur|евро)/i.test(userText);
  const mentionsPermit = /(разрешен(ие|я)\s+на\s*работ|work\s*permit|zaměstnanecká|povolen[ií])/i.test(userText);
  const mentionsSlots  = /(слот|очеред|термин|запис)/i.test(userText);
  const mentionsPay    = /(оплат|сч[её]т|инвойс|банк|pay|invoice)/i.test(userText);

  const poolBudget = ['Честно, для меня это сейчас дорого.', 'Пока не готов закрывать всю сумму.'];
  const poolAfterPermit = ['Предпочитаю оплату после визы или хотя бы подтверждения регистрации.'];
  const poolSlots = ['Сначала запись/подтверждение, потом вернусь к оплате.'];
  const poolDelay = ['Возьму время на внутреннюю проверку и подберу кандидатов.'];

  // Новое: без явных триггеров и не на этапе Payment — не возражаем
  const hasTrigger = hasPriceTalk || mentionsPermit || mentionsSlots || mentionsPay || stage === 'Payment';
  if (!hasTrigger) return null;

  let chosen = '';
  if (hasPriceTalk)        chosen = poolBudget[Math.floor(rnd()*poolBudget.length)];
  else if (mentionsPermit) chosen = poolAfterPermit[0];
  else if (mentionsSlots)  chosen = poolSlots[0];
  else                     chosen = poolDelay[Math.floor(rnd()*poolDelay.length)];

  if (S.lastObjection && S.lastObjection.toLowerCase() === chosen.toLowerCase()) {
    chosen = 'Давайте аккуратно, без лишних рисков.';
  }
  S.lastObjection = chosen;

  let stageSuggestion = 'Contract';
  if (!hasDemand) stageSuggestion = 'Demand';
  else if (!hasCoop) stageSuggestion = 'Contract';
  else if (uniqEvidence < 2) stageSuggestion = 'Candidate';
  else stageSuggestion = 'Payment';

  return { text: chosen, stage: stageSuggestion };
}

function postRules({ parsed, trust, evidences, history, userText, sid, evidenceDetails }) {
  const S = getState(sid);
  S.turn = (S.turn || 0) + 1;

  // 0) Нормализуем вход: редиректор «контракт работодателя» → B2B
  userText = redirectEmployerContractToCoop(userText);

  const inc = new Set((evidences || []).filter(k => !S.seenEvidences.has(k)));

  let reply = String(parsed.reply || '').trim();
  const setActions = new Set(parsed.suggestedActions || []);

  // Быстрый ответ «кто ты?»
  if (/(как.*зовут|вас зовут|ваше имя|who are you)/i.test(userText)) {
    reply = 'Меня зовут Али.';
    parsed.stage ??= 'Greeting';
  }

  // Регистрации/слоты — коротко по делу
  if (/(когда|последн).*(регистрир|записыва)|слот|очеред/i.test(userText)) {
    reply = registrationAnswer();
    parsed.stage = 'Demand';
    parsed.needEvidence = false;
  }

  // Greeting — коротко и тепло, если собеседник представился
  const isEarly = (history || []).length <= 2 || (!parsed.stage || parsed.stage === 'Greeting');
  if (isEarly) {
    reply = craftHumanGreeting({ base: reply, userText });
    parsed.stage = 'Greeting';
  } else {
    reply = rewriteVacancyQuestionToSupplierRole(reply);
  }

  // DEMAND-факты — отвечаем только если менеджер спросил
  const DF = getDemandFacts(sid);
  const askedSalary = /(зарплат|salary|сколько.*(получ|net))/i.test(userText);
  const askedHouse  = /(жиль|accommodat|общежит|проживан)/i.test(userText);
  const askedHours  = /(график|час(ов)?\s*в\s*месяц|смен|working\s*hours|work\s*time)/i.test(userText);
  const askedLoc    = /(локац|город|место|location|where)/i.test(userText);
  const askedWhatJob= /(что\s+делать|обязан|описани[ея]\s+работ|job\s*description)/i.test(userText);
  if (Object.keys(DF).length) {
    if (askedSalary) { reply = formatFactsShort(DF,'salary') || reply; parsed.stage ??= 'Demand'; }
    else if (askedHouse) { reply = formatFactsShort(DF,'accommodation') || reply; parsed.stage ??= 'Demand'; }
    else if (askedHours) { reply = formatFactsShort(DF,'hours') || reply; parsed.stage ??= 'Demand'; }
    else if (askedLoc)   { reply = formatFactsShort(DF,'location') || reply; parsed.stage ??= 'Demand'; }
    else if (askedWhatJob) { reply = formatFactsShort(DF,'all') || reply; parsed.stage ??= 'Demand'; }
  }

  // Реактивная просьба документов — только если менеджер спросил «что нужно»
  if (/(что\s+нужно|what.*need|какие\s+документ\w*\s+нужн)/i.test(userText)) {
    reply = 'Обычно достаточно описания вакансии (Demand) и нашего B2B-контракта.';
    parsed.stage = hasEvidence(sid,'demand_letter') ? 'Contract' : 'Demand';
    parsed.needEvidence = true;
    setActions.add('ask_demands');
    setActions.add('ask_coop_contract');
  }

  // Тихо фиксируем материалы (без «спасибо, получил»)
  if (inc.has('business_card') || (evidenceDetails && evidenceDetails.business_card)) {
    bumpEvidence(sid, 'business_card', evidenceDetails?.business_card);
  }
  if (inc.has('demand_letter')) {
    bumpEvidence(sid, 'demand_letter');
    const facts = extractDemandFactsFromDetails(evidenceDetails || {});
    if (Object.keys(facts).length) setDemandFacts(sid, facts);

    if (!hasEvidence(sid,'coop_contract_pdf')) {
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      setActions.add('ask_coop_contract');
    }
  }
  if (inc.has('sample_contract_pdf')) {
    bumpEvidence(sid, 'sample_contract_pdf');
    if (!hasEvidence(sid,'coop_contract_pdf')) {
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      setActions.add('ask_coop_contract');
    }
  }
  if (inc.has('coop_contract_pdf')) {
    bumpEvidence(sid, 'coop_contract_pdf');
    parsed.stage = 'Contract';
    parsed.needEvidence = false;
  }
  for (const key of ['visa_sample','presentation','video','website','company_registry','reviews','registry_proof','price_breakdown','slot_plan','invoice_template','nda']) {
    if (inc.has(key)) bumpEvidence(sid, key, evidenceDetails?.[key]);
  }

  // Оплата — только реактивно и без призывов
  if (/(банк|банковск|crypto|крипто|usdt|btc|eth|криптовалют)/i.test(userText) && /оплат|плат[её]ж|инвойс|сч[её]т/i.test(userText)) {
    reply = 'Предпочитаю оплату после визы или как минимум после подтверждения регистрации. Крипту не люблю, счёт — банковский.';
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  // Возражения при низком доверии — только при триггерах
  const uniqEvidence = evidenceCountUnique(sid);
  const hasDemandEv = hasEvidence(sid,'demand_letter');
  const hasCoopEv   = hasEvidence(sid,'coop_contract_pdf');
  if ((parsed.stage === 'Payment' && trust < 90) || /(цена|дорог|оплат|сч[её]т|инвойс)/i.test(userText)) {
    const obj = chooseObjection({
      sid, userText, trust, uniqEvidence,
      hasDemand: hasDemandEv, hasCoop: hasCoopEv,
      stage: parsed.stage
    });
    if (obj) {
      reply = obj.text;
      parsed.stage = obj.stage;
      parsed.needEvidence = !hasDemandEv || !hasCoopEv || uniqEvidence < 2;
      if (!hasDemandEv) setActions.add('ask_demands');
      if (!hasCoopEv) setActions.add('ask_coop_contract');
    }
  }

  // Если модель случайно вставила цены — вырезаем
  reply = reply.replace(/(?:€|\$\s?|\b(czk|eur|usd)\b)[\s:]*[\d\s.,]+/gi, '').trim();

  // Санитария
  reply = redirectEmployerContractToCoop(reply);
  reply = stripEmployerRequisitesRequests(reply);
  reply = stripRequisitesFromDemand(reply);
  reply = stripRoboticAcks(reply);
  reply = cleanSales(reply);
  reply = stripSalesy(reply);
  reply = fixGuaranteeStoryText(reply);
  reply = forceMasculine(reply);
  reply = limitSentences(reply, MAX_SENTENCES);

  // Анти-луп
  if (reply && S.lastReply && reply.toLowerCase() === S.lastReply.toLowerCase()) {
    reply = 'Ок.';
  }

  // Анти-повторы/кулдауны
  reply = repetitionGuard(reply, sid);
  S.lastReply = reply;

  // Финальные ворота: даже при высоком доверии НЕ инициируем «инвойс/счёт»
  const gatesOk = (trust >= 90 && evidenceCountUnique(sid) >= 2 && hasEvidence(sid,'coop_contract_pdf'));
  if (gatesOk) {
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
    // не добавляем invoice_request — инициативы нет
  }

  parsed.reply = reply.trim();
  parsed.suggestedActions = normalizeActions(Array.from(setActions));

  if (/(demand|контракт|документ|полный контракт|сотрудничеств)/i.test(parsed.reply) && (!parsed.stage || parsed.stage === 'Greeting')) {
    parsed.stage = 'Demand';
  }

  return parsed;
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 5.2. УСЛОВИЯ ПОКУПКИ И ПОЗИЦИЯ АЛИ
   ────────────────────────────────────────────────────────────── */

function applyAliPurchasePolicy({
  reply = '',
  stage = 'Greeting',
  trust = 0,
  evidences = [],
  userText = '',
  sid = 'default',
  hasDemandEv = false,
  hasCoopEv = false
} = {}) {
  const rx = {
    pressure: /(срочн|прямо сейчас|немедленн|today|сегодня|within\s*24|24\s*час)/i,
    prepay: /(предоплат|аванс|оплатить\s*до|до\s*визы|advance\s*payment)/i,
    cryptoOnly: /(только\s*крипт|only\s*crypto)/i,
    discount: /(скидк|дешевле|сбросьте\s*цену|дискаунт)/i,
    unrealDeadline: /(2\s*дня|48\s*час|за\s*два\s*дня|через\s*сутки)/i,
    payBeforeDocs: /(оплат.*(до|без)\s*(документ|контракт|demand|регистрац))/i
  };

  const baseByStage = {
    Greeting:  'Давайте начнём с сути вопроса.',
    Demand:    'Сначала описание вакансии и проверяемость.',
    Contract:  'Хочу видеть договорённости на бумаге.',
    Candidate: 'Готов двигаться аккуратно.',
    Payment:   'Предпочитаю банковский счёт.',
    Closing:   'Окей, на связи.'
  };

  const rebuttals = [];
  if (rx.pressure.test(userText)) rebuttals.push('Мне некомфортно при давлении.');
  if (rx.prepay.test(userText) || rx.payBeforeDocs.test(userText)) rebuttals.push('Предоплату не рассматриваю. Сначала подтверждения по документам.');
  if (rx.cryptoOnly.test(userText)) rebuttals.push('Крипту не люблю. Если дойдём до оплаты — предпочту банковский счёт.');
  if (rx.discount.test(userText)) rebuttals.push('Торг тут не главное — важнее прозрачность и проверяемость.');
  if (rx.unrealDeadline.test(userText)) rebuttals.push('Сроки «за два дня» звучат нереалистично.');

  let needEvidence = undefined;
  if (stage === 'Payment' && trust < 90) {
    rebuttals.push('Нужна уверенность в процессе. Платежи — после подтверждений.');
    needEvidence = needEvidence ?? (!hasDemandEv || !hasCoopEv);
  }

  const merged = joinUniqueSentences([baseByStage[stage] || '', rebuttals.join(' ')]).trim();
  if (!merged) return null;

  return { reply: merged, stage, needEvidence, actions: [] };
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 5.3. АЛИ ПОКУПАЕТ (вероятности, условия, крипта/банк)
   ────────────────────────────────────────────────────────────── */

function evaluateObjectionHandling(text=''){
  const t = String(text).toLowerCase();
  let score = 0;
  const pats = [
    /(понимаю|не\s*настаиваю|спокойно|без\s*давления)/i,
    /(вопрос\s*не\s*в\s*цене|ценност|репутац|долгосроч)/i,
    /(начн[её]м?\s*с\s*одн(ого|ого\s*клиент|ого\s*кандид))/i,
    /(проверьте\s*работу|проверить\s*работу|мисси|партнер|партнёр|над[её]жн)/i,
    /(не\s*отвечайте\s*сейчас|как\s*будете\s*готовы)/i
  ];
  for (const r of pats) if (r.test(t)) score++;
  let level = 'none', bonus = 0;
  if (score >= 4) { level = 'strong'; bonus = 0.10; }
  else if (score >= 2) { level = 'weak'; bonus = 0.03; }
  return { level, bonus, score };
}

function evaluateCryptoPitch(text=''){
  const t = String(text).toLowerCase();
  let score = 0;
  if (/(4-?7|4\s*–\s*7|4\s*до\s*7)\s*(рабочих\s*)?дн/i.test(t) || /bank.*(4|four).*(days|дн)/i.test(t)) score++;
  if (/(5\s*мин|5\s*minutes|в\s*течение\s*5\s*мин)/i.test(t)) score++;
  if (/(начать\s*сразу|незамедлительно|faster|быстрее|скорост|ускорит)/i.test(t)) score++;
  let level = 'none';
  if (score >= 2) level = 'strong';
  else if (score === 1) level = 'weak';
  return { level, score };
}

function decideCryptoAcceptance({ trust=0, cryptoFx={level:'none'}, rnd = Math.random } = {}){
  let base = 0;
  if (trust >= 100) base = 0.45;
  else if (trust >= 90) base = 0.30;
  else if (trust >= 80) base = 0.15;
  else base = 0.05;

  let bonus = 0;
  if (cryptoFx.level === 'strong') bonus = 0.15;
  else if (cryptoFx.level === 'weak') bonus = 0.05;

  const p = Math.max(0, Math.min(0.75, base + bonus));
  return rnd() < p;
}

function chooseCandidateCount(rnd = Math.random){
  const r = rnd();
  if (r < 0.50) return 1;
  if (r < 0.75) return 2;
  if (r < 0.85) return 3;
  if (r < 0.90) return 4;
  if (r < 0.94) return 5;
  if (r < 0.97) return 6 + Math.floor(rnd()*2); // 6–7
  return 8 + Math.floor(rnd()*3); // 8–10
}

function pluralRu(n, one, few, many){
  const n10 = n % 10, n100 = n % 100;
  if (n10 === 1 && n100 !== 11) return one;
  if (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) return few;
  return many;
}

function applyAliPurchaseDecision({
  reply = '',
  stage = 'Greeting',
  trust = 0,
  evidences = [],
  userText = '',
  sid = 'default'
} = {}) {
  const S = getState(sid);
  if (S.alreadyCommitted) return null;

  const hasCard   = hasEvidence(sid, 'business_card');
  const hasDemand = hasEvidence(sid, 'demand_letter');
  const hasSample = hasEvidence(sid, 'sample_contract_pdf');
  const hasCoop   = hasEvidence(sid, 'coop_contract_pdf');

  const prereqsOk = hasCard && hasDemand && hasSample && hasCoop && trust >= 70;
  if (!prereqsOk) return null;

  function baseProbByTrust(t){
    if (t >= 100) return 0.50;
    if (t >= 90)  return 0.35;
    if (t >= 80)  return 0.05;
    return 0.01; // t ∈ [70..79]
  }

  const objectionFx = evaluateObjectionHandling(userText);
  const cryptoFx = evaluateCryptoPitch(userText);

  let pBuy = baseProbByTrust(trust) + objectionFx.bonus;
  pBuy = Math.max(0, Math.min(0.85, pBuy));

  const rnd = seededRand(`${sid}#buy#${getState(sid).turn || 0}`);
  const willBuy = rnd() < pBuy;
  if (!willBuy) return null;

  const candidates = chooseCandidateCount(rnd);
  const wantsCrypto = decideCryptoAcceptance({ trust, cryptoFx, rnd });

  let buyLine = `Я готов с вами сотрудничать. Стартуем с ${candidates} кандидат${pluralRu(candidates, 'ом','ами','ами')}. `;
  buyLine += wantsCrypto
    ? 'Предоставьте, пожалуйста, криптовалютные реквизиты для оплаты.'
    : 'Предоставьте, пожалуйста, банковский счёт для оплаты.';

  S.alreadyCommitted = true;
  return {
    reply: buyLine,
    stage: 'Payment',
    needEvidence: false,
    actions: ['invoice_request']
  };
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 5.4. LLM-ОРКЕСТРАТОР (runLLM)
   ────────────────────────────────────────────────────────────── */

async function runLLM({ history, message, evidences, stage, sessionId='default', evidenceDetails }) {
  const trust = computeTrust({
    baseTrust: 20,
    evidences: Array.from(new Set(evidences || [])),
    history: history || [],
    lastUserText: message || ''
  });

  const safeMessage = redirectEmployerContractToCoop(message || '');

  const messages = buildMessages({
    history, message: safeMessage, trust, evidences, evidenceDetails
  });

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
  const json = extractFirstJsonObject(raw) || { reply: '' };

  let parsed;
  try {
    parsed = LLMShape.parse(json);
  } catch {
    parsed = null;
  }

  if (!parsed) {
    const fb = stage === 'Payment'
      ? 'Мне важны подтверждения по документам.'
      : 'Опишите, пожалуйста, предложение или пришлите документы.';
    parsed = {
      reply: fb,
      confidence: Math.max(0, Math.min(60, trust)),
      stage: stage || 'Greeting',
      needEvidence: false,
      suggestedActions: []
    };
  }

  parsed.reply = String(parsed.reply || '').trim();
  parsed.stage = String(parsed.stage || stage || 'Greeting');
  parsed.confidence = Math.max(0, Math.min(100, Number(parsed.confidence ?? trust)));
  parsed.needEvidence = Boolean(parsed.needEvidence);
  parsed.suggestedActions = Array.isArray(parsed.suggestedActions) ? parsed.suggestedActions : [];

  parsed = postRules({
    parsed,
    trust,
    evidences,
    history,
    userText: safeMessage,
    sid: sessionId || 'default',
    evidenceDetails
  });

  return { trust, evidenceCount: evidenceCountUnique(sessionId), result: parsed };
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 6. РОУТЫ: root/assets, API, совместимость
   ────────────────────────────────────────────────────────────── */

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

app.get('/api/ping', (_, res) => res.json({ ok: true }));

app.get('/api/version', (_,res) => res.json({ ok:true, name:'renovogo-llm-backend', version:'2025-09-16-11' }));

function sanitizeHistory(arr){
  return Array.isArray(arr) ? arr.slice(-50).map(h => ({
    role: (h.role === 'assistant' ? 'assistant' : 'user'),
    content: String(h.content || '').replace(/<[^>]+>/g, ''),
    stage: h.stage ? String(h.stage) : undefined
  })) : [];
}

function normalizeEvidenceKey(k){
  const key = String(k || '').toLowerCase().trim();
  const map = new Map([
    ['card','business_card'], ['визитка','business_card'], ['business_card','business_card'],
    ['demand','demand_letter'], ['demandletter','demand_letter'], ['деманд','demand_letter'],
    ['sample','sample_contract_pdf'], ['sample_contract','sample_contract_pdf'],
    ['contract_sample','sample_contract_pdf'], ['пример_контракта','sample_contract_pdf'],
    ['contract_pdf','coop_contract_pdf'], ['contract','coop_contract_pdf'], ['contractpdf','coop_contract_pdf'], ['договор','coop_contract_pdf'],
    ['coop_contract','coop_contract_pdf'], ['full_contract','coop_contract_pdf'], ['контракт_о_сотрудничестве','coop_contract_pdf'],
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

/* /api/reply */
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

/* /api/score — смягчённый ранний анализ */
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
    const msgText = history.filter(h => h.role === 'user').map(h => h.content || '').join('\n');

    const trust = computeTrust({ baseTrust: 20, evidences, history, lastUserText });
    const early = history.length < 6; // ранняя стадия беседы

    const good = [];
    const bad  = [];

    // Позитивы
    if (/(здрав|прив|добрый)/i.test(msgText)) good.push('Вежливое приветствие');
    if (/renovogo|renovogo\.com/i.test(msgText)) good.push('Дали проверяемый факт');
    if (evidences.length >= 2) good.push('Приложили ≥2 доказательства');
    if (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText)) good.push('Есть финальный CTA');

    // Рекомендации — мягче в начале диалога
    if (trust < 80 && !early) {
      bad.push('Для предметного обсуждения добавьте документы (Demand/Contract/Registry).');
    }
    if (/(оплат|сч[её]т|инвойс|цен|стоим|€|eur|czk)/i.test(msgText)) {
      bad.push('Не смешивайте сервисные платежи с зарплатой — это разные вещи.');
    }

    // Балл: гарантируем нижний порог вежливости
    const baseCourtesy = /(здрав|прив|добрый)/i.test(msgText) ? 15 : 0;
    const baseFact = /renovogo|renovogo\.com/i.test(msgText) ? 15 : 0;
    const docPts = (evidences.length >= 2) ? 35 : 0;
    const ctaPts = (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText) ? 35 : 0);

    const final = clamp(Math.round(baseCourtesy + baseFact + docPts + ctaPts), 10, 100); // минимум 10

    res.json({ final, good, bad, trust, evidences: evidences.length });
  } catch (e) {
    logError(e, '/api/score');
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

/* Совместимость со старым роутом */
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

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 7. СТАРТ
   ────────────────────────────────────────────────────────────── */

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LLM backend running on :${PORT}`));
