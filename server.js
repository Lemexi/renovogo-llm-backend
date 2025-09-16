// server.js — RenovoGo LLM backend (stable memory, no robotic ACKs, realistic flow)
// v2025-09-16-9

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
    demandFacts: Record<string, any>
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
      demandFacts: Object.create(null)
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
  return `По долгосрочному — ${REG_LONGTERM_MONTHS} мес назад; по сезонному — ${REG_SEASONAL_MONTHS} мес назад. Очереди нестабильные. Сначала проверю Demand Letter и контракт.`;
}

// greeting helpers
function extractUserName({ history = [], evidenceDetails = {} } = {}) {
  const cardName = evidenceDetails?.business_card?.name;
  if (cardName && String(cardName).trim()) return String(cardName).trim();
  const lastUser = [...(history||[])].reverse().find(h => h.role === 'user');
  const txt = (lastUser?.content || '').trim();
  const m = txt.match(/(?:я|меня\s+зовут|это)\s+([A-ZА-ЯЁ][a-zа-яё\-\']{1,30})/i);
  if (m) return m[1];
  return '';
}
function rewriteVacancyQuestionToSupplierRole(text=''){
  let t = String(text);
  t = t.replace(/какие\s+вакансии[^.?!]*ищете[^.?!]*[.?!]?/gi, 'Какие вакансии у вас доступны?');
  t = t.replace(/мы\s+работаем[^.?!]*чех[иия][^.?!]*польш[^.?!]*уже\s+давно[^.?!]*[.?!]?/gi, '');
  t = t.replace(/\s{2,}/g, ' ').replace(/\s+([?.!])/g, '$1').trim();
  return t;
}
function craftHumanGreeting({ base='', name='' } = {}){
  const variants = name
    ? [
        `Привет, ${name}. Чем могу помочь?`,
        `${name}, рад знакомству. С какими задачами ко мне пришли?`,
        `${name}, мне тоже приятно. Какие вакансии у вас сейчас открыты?`
      ]
    : [
        'Привет! Чем могу помочь?',
        'Рад знакомству. Какие вакансии у вас сейчас открыты?',
        'Здравствуйте. С чего начнём: какие вакансии у вас доступны?'
      ];
  let t = rewriteVacancyQuestionToSupplierRole(base);
  if (!t || t.length < 3) {
    t = variants[Math.floor(Math.random()*variants.length)];
  } else if (!/[?]$/.test(t) && !/ваканси/i.test(t)) {
    t = `${t} Какие вакансии у вас доступны?`;
  }
  return forceMasculine(t);
}

/* ──────────────────────────────────────────────────────────────
   ЧАСТЬ 5. LLM: сборка сообщений, ретраи, пост-правила
   ────────────────────────────────────────────────────────────── */

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
      `Запрещено требовать "контракт работодателя" или "реквизиты работодателя" — нужен ТОЛЬКО наш B2B-контракт. ` +
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

/** Жёсткий редиректор любых формулировок «контракта/реквизитов работодателя» в «наш B2B контракт» */
function redirectEmployerContractToCoop(text=''){
  let t = String(text || '');
  // Явные формулировки «контракт работодателя / employer contract»
  t = t.replace(
    /\b(контракт(?!\s*о\s*сотрудничестве)[^.!?\n]*\b(работодател[ьяею]|employer|company)[^.!?\n]*)/gi,
    'контракт о сотрудничестве (наш B2B)'
  );
  // Любые «реквизиты работодателя / employer details»
  t = t.replace(
    /\b(реквизит\w*\s*(работодател[ьяею]|employer|company)|employer\s*(details|requisites))\b/gi,
    'реквизиты для инвойса (наши B2B)'
  );
  // Смягчаем косвенные упоминания «документы работодателя»
  t = t.replace(
    /\b(документ\w*\s*(работодател[ьяею]|employer|company))\b/gi,
    'B2B-документы по сотрудничеству'
  );
  return t;
}

// «человечные» возражения на низком доверии
function chooseObjection({ sid, userText='', trust=0, uniqEvidence=0, hasDemand=false, hasCoop=false }){
  const S = getState(sid);
  const rnd = seededRand(sid);
  const hasPriceTalk   = /(цена|стоим|дорог|price|€|eur|евро)/i.test(userText);
  const mentionsPermit = /(разрешен(ие|я)\s+на\s*работ|work\s*permit|zaměstnanecká|povolen[ií])/i.test(userText);
  const mentionsSlots  = /(слот|очеред|термин|запис)/i.test(userText);

  const poolBudget = [
    'Честно, для меня это сейчас дорого. Давайте начнём с одного кандидата и закрепим результат.',
    'Пока не готов закрывать всю сумму — покажите, как пройдёт первый кейс, и двинемся дальше.',
    'Дорого выходит на старте. Снизим риски: один кандидат, остальное — после подтверждения хода дела.'
  ];
  const poolAfterPermit = [
    'Готов оплатить после разрешения на работу или хотя бы подтверждения регистрации.',
    'Оплачу, когда увижу разрешению на работу — до этого хочу минимизировать риски.',
    'Как только будет подтверждение по регистрации в посольство, перейду к оплате без задержек.'
  ];
  const poolSlots = [
    'Вижу, что с регестрацией в посольство нестабильно сейчас. Предлагаю стартовать с 1 кандидатом без предоплаты, оплачу после визы.',
    'Давайте зафиксируем двух кандидат, а оплату расширим после того как получу от Вас PDF файл.',
    'Сначала запись — потом увеличим объём и финансирование. Так будет безопаснее.'
  ];
  const poolGeneric = [
    'Давайте начнём осторожно: один кандидат, остальное — после промежуточной проверки.',
    'Мне важно увидеть первый успешный кейс. После него готов масштабироваться и обсуждать оплату.',
    'Готов продолжить, но пока без полной оплаты. Покажите движение по документам — и закрываем счёт.'
  ];
  const poolDelay = [
    'Возьму неделю на внутреннее согласование и подбор вам персонала. Напишу вам в этот же чат.',
    'Нужно до 5 рабочих дней на проверку и сбор кандидатов. Вернусь с апдейтом.',
    'Поставлю себе напоминание на следующую неделю и вернусь к вам.'
  ];

  let chosen = '';
  if (hasPriceTalk)        chosen = poolBudget[Math.floor(rnd()*poolBudget.length)];
  if (!chosen && mentionsPermit) chosen = poolAfterPermit[Math.floor(rnd()*poolAfterPermit.length)];
  if (!chosen && mentionsSlots)  chosen = poolSlots[Math.floor(rnd()*poolSlots.length)];
  if (!chosen && uniqEvidence < 2) chosen = poolGeneric[Math.floor(rnd()*poolGeneric.length)];
  if (!chosen && trust < 60)       chosen = poolDelay[Math.floor(rnd()*poolDelay.length)];
  if (!chosen)                      chosen = poolGeneric[Math.floor(rnd()*poolGeneric.length)];

  if (S.lastObjection && S.lastObjection.toLowerCase() === chosen.toLowerCase()) {
    const alt = [...poolGeneric, ...poolBudget, ...poolAfterPermit, ...poolSlots, ...poolDelay]
      .filter(x => x.toLowerCase() !== chosen.toLowerCase());
    if (alt.length) chosen = alt[Math.floor(rnd()*alt.length)];
  }
  S.lastObjection = chosen;

  let nudge = '';
  if (!hasDemand) nudge = 'Скиньте Demand Letter с описанием вакансий — от него оттолкнёмся.';
  else if (!hasCoop) nudge = 'Нужен полный контракт о сотрудничестве B2B — проверю условия и вернусь по оплате.';
  else if (uniqEvidence < 2) nudge = 'Сейчас нет свободных кандидатов — вернусь с подборкой через неделю.';
  else nudge = 'Начнём с одного кандидата и параллельно согласуем оплату по этапам.';

  let stageSuggestion = 'Contract';
  if (!hasDemand) stageSuggestion = 'Demand';
  else if (!hasCoop) stageSuggestion = 'Contract';
  else if (uniqEvidence < 2) stageSuggestion = 'Candidate';
  else stageSuggestion = 'Payment';

  return { text: joinUniqueSentences([chosen, nudge]), stage: stageSuggestion };
}

// пост-правила
function postRules({ parsed, trust, evidences, history, userText, sid, evidenceDetails }) {
  const S = getState(sid);

  // 0) Нормализуем вход пользователя: насильно уводим от «контракта работодателя»
  userText = redirectEmployerContractToCoop(userText);

  const inc = new Set((evidences || []).filter(k => !S.seenEvidences.has(k)));

  let reply = String(parsed.reply || '').trim();
  const setActions = new Set(parsed.suggestedActions || []);

  // Быстрый ответ «кто ты?»
  if (/(как.*зовут|вас зовут|ваше имя|who are you)/i.test(userText)) {
    reply = 'Меня зовут Али.';
    parsed.stage ??= 'Greeting';
  }

  // Регистрации/слоты
  if (/(когда|последн).*(регистрир|записыва)|слот|очеред/i.test(userText)) {
    reply = registrationAnswer();
    parsed.stage = 'Demand';
    parsed.needEvidence = true;
    parsed.suggestedActions = ['ask_demands','ask_coop_contract'];
  }

  // Greeting rewrite
  const isEarly = (history || []).length <= 2 || (!parsed.stage || parsed.stage === 'Greeting');
  if (isEarly) {
    const userName = extractUserName({ history, evidenceDetails });
    reply = craftHumanGreeting({ base: reply, name: userName });
    parsed.stage = 'Greeting';
  } else {
    reply = rewriteVacancyQuestionToSupplierRole(reply);
  }

  // Ответы на базовые вопросы из DEMAND
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
  // Изредка уточняем один пункт (12%)
  const rndProbe = seededRand(sid)();
  if (Object.keys(DF).length && rndProbe < 0.12) {
    if (!DF.salary?.value) reply = joinUniqueSentences([reply, 'Уточню: нетто ставка по этой позиции какая?']);
    else if (!DF.hours_monthly && !DF.schedule) reply = joinUniqueSentences([reply, 'Уточню график: сколько часов в месяц/смены?']);
    else if (!DF.accommodation?.cost_eur) reply = joinUniqueSentences([reply, 'Жильё: напомните ориентир по стоимости?']);
  }

  // Вопрос про количество кандидатов
  const askCandidates = /(сколько.*кандидат|кандидат(ов)?\s*(есть|готов|подад)|сколько человек будем подавать)/i.test(userText);
  if (askCandidates) {
    if (hasEvidence(sid,'demand_letter') && hasEvidence(sid,'coop_contract_pdf')) {
      reply = 'Готов подать 1–2 кандидата на старт в Чехию. Остальные — после регистрации в посольство.';
      parsed.stage = parsed.stage === 'Payment' ? 'Payment' : 'Candidate';
      parsed.needEvidence = false;
      setActions.add('invoice_request');
    } else {
      const miss = [];
      if (!hasEvidence(sid,'demand_letter')) miss.push('Demand Letter');
      if (!hasEvidence(sid,'coop_contract_pdf')) miss.push('полный контракт о сотрудничестве');
      reply = `Подтвержу точное число сразу после ${miss.join(' и ')}. Сейчас ориентир — 1–2 кандидата на старт.`;
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      parsed.suggestedActions = ['ask_demands','ask_coop_contract'];
    }
  }

  // Тихая фиксация материалов
  if (inc.has('business_card') || (evidenceDetails && evidenceDetails.business_card)) {
    bumpEvidence(sid, 'business_card', evidenceDetails?.business_card);
  }
  if (inc.has('demand_letter')) {
    bumpEvidence(sid, 'demand_letter');
    const facts = extractDemandFactsFromDetails(evidenceDetails || {});
    if (Object.keys(facts).length) setDemandFacts(sid, facts);

    if (!hasEvidence(sid,'coop_contract_pdf')) {
      setActions.add('ask_coop_contract');
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      reply = joinUniqueSentences([reply,
        'Запрос вижу. Направление сложное: по Индии сейчас с регестрацией в посольство вобше нестабильно. Начнём с одного кандидата, остальные — после подтверждения записи.'
      ]);
    } else {
      reply = joinUniqueSentences([reply, formatFactsShort(getDemandFacts(sid), 'all') || '']);
    }
  }
  if (inc.has('sample_contract_pdf')) {
    bumpEvidence(sid, 'sample_contract_pdf');
    if (!hasEvidence(sid,'coop_contract_pdf')) {
      setActions.add('ask_coop_contract');
      parsed.stage = 'Contract';
      parsed.needEvidence = true;
      reply = joinUniqueSentences([reply, 'Отправьте, пожалуйста, полный договор B2B.']);
    }
  }
  if (inc.has('coop_contract_pdf')) {
    bumpEvidence(sid, 'coop_contract_pdf');
    parsed.stage = 'Contract';
    parsed.needEvidence = false;
    setActions.add('ask_price_breakdown');
    reply = joinUniqueSentences([reply, 'Спасибо, как изучу вернусь к вам.']);
  }
  for (const key of ['visa_sample','presentation','video','website','company_registry','reviews','registry_proof','price_breakdown','slot_plan','invoice_template','nda']) {
    if (inc.has(key)) bumpEvidence(sid, key, evidenceDetails?.[key]);
  }

  // Политика оплаты — только банк
  if (/(банк|банковск|crypto|крипто|usdt|btc|eth|криптовалют)/i.test(userText) && /оплат|плат[её]ж|инвойс|сч[её]т/i.test(userText)) {
    reply = 'Банковский инвойс. Криптовалюту не принимаем.';
    setActions.add('invoice_request');
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  // Возражения при низком доверии
  const uniqEvidence = evidenceCountUnique(sid);
  const hasDemand = hasEvidence(sid,'demand_letter');
  const hasCoop   = hasEvidence(sid,'coop_contract_pdf');
  if (parsed.stage === 'Payment' && trust < 90) {
    const obj = chooseObjection({ sid, userText, trust, uniqEvidence, hasDemand, hasCoop });
    reply = obj.text;
    parsed.stage = obj.stage;
    parsed.needEvidence = !hasDemand || !hasCoop || uniqEvidence < 2;
    if (!hasDemand) setActions.add('ask_demands');
    if (!hasCoop) setActions.add('ask_coop_contract');
    if (hasCoop) setActions.add('ask_price_breakdown');
  }

  // Если ответ снова спрашивает базовые вещи, а факты есть — подменим на факты
  if (Object.keys(DF).length) {
    if (/какая\s+зарплат/i.test(reply)) reply = formatFactsShort(DF,'salary') || reply;
    if (/есть\s+ли\s+жиль|жиль[её]\s+будет|стоимость\s+жиль/i.test(reply)) reply = formatFactsShort(DF,'accommodation') || reply;
    if (/график|сколько\s+час/i.test(reply)) reply = formatFactsShort(DF,'hours') || reply;
    if (/какая\s+локац|где\s+работ/i.test(reply)) reply = formatFactsShort(DF,'location') || reply;
  }

  // Санитария (+ жёсткий редиректор на наш B2B)
  reply = redirectEmployerContractToCoop(reply);
  reply = stripEmployerRequisitesRequests(reply);
  reply = stripRequisitesFromDemand(reply);
  reply = stripRoboticAcks(reply);
  reply = cleanSales(reply);
  reply = forceMasculine(reply);
  reply = limitSentences(reply, MAX_SENTENCES);

  // Анти-луп
  if (reply && S.lastReply && reply.toLowerCase() === S.lastReply.toLowerCase()) {
    reply = 'Предлагаю стартовать с одного кандидата. Документы проверю и дальше будем работать по прайсу.';
    parsed.stage = hasCoop ? 'Payment' : 'Contract';
    if (!hasCoop) { setActions.add('ask_coop_contract'); parsed.needEvidence = true; }
    setActions.add('test_one_candidate');
  }
  S.lastReply = reply;

  // Финальные ворота
  const gatesOk = (trust >= 90 && evidenceCountUnique(sid) >= 2 && hasEvidence(sid,'coop_contract_pdf'));
  if (gatesOk) {
    setActions.add('invoice_request');
    if (!/инвойс|сч[её]т|реквизит/i.test(reply)) {
      reply = conciseJoin([reply, 'Готов перейти к финализации — пришлите реквизиты для оплаты.']);
    }
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
  }

  parsed.reply = reply.trim();
  parsed.suggestedActions = normalizeActions(Array.from(setActions));

  if (/(demand|контракт|документ|полный контракт|сотрудничеств)/i.test(parsed.reply) && (!parsed.stage || parsed.stage === 'Greeting')) {
    parsed.stage = 'Demand';
  }

  return parsed;
}

async function runLLM({ history, message, evidences, stage, sessionId='default', evidenceDetails }) {
  const trust = computeTrust({
    baseTrust: 20,
    evidences: Array.from(new Set(evidences || [])),
    history: history || [],
    lastUserText: message || ''
  });

  // Входной текст также нормализуем (никаких «контрактов работодателя»)
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
  const json = extractFirstJsonObject(raw);

  let parsed;
  try { parsed = LLMShape.parse(json); } catch { parsed = null; }

  if (!parsed) {
    const fb = stage === 'Payment'
      ? 'Начнём осторожно: один кандидат, остальное — после промежуточной проверки.'
      : 'Нужны Demand Letter и контракт о сотрудничестве. После проверки обсудим сроки и цену.';
    parsed = {
      reply: fb,
      confidence: clamp(trust, 0, 60),
      stage: stage || 'Demand',
      needEvidence: true,
      suggestedActions: ['ask_demands','ask_coop_contract']
    };
  }

  // типобезопасность
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

/* /api/score */
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

/* совместимость со старым роутом */
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
