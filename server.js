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
  .split(',').map(s => s.trim()).filter(Boolean);

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

// ---------- PRICEBOOK (в контекст без «училок») ----------
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

// ---------- Small utils ----------
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

// ---------- «живые» реплики по темам ----------
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
  "Вы сначала о вакансии мне раскажите",
  "У вас есть писмьо запрос, или вы так просто решили поговорить?",
  "Покажите мне контракт, я его проверю, тогда поговорим",
  "Вы не профисионал своего дела, что за вакансия, сколько платят, куда требуется кандидат?",
  "Я не первый день в этом бизнесе, вы мне еще нечего не расказали а уже что-то требуете."
];

const TOO_EARLY_PAY_HUMAN = [
  "Какая оплата, вы что с дуба рухнули? Я вас впервые вижу.",
  "Серьёзно? Мы ещё ничего не проверили, а вы уже про деньги.",
  "Сейчас речи об оплате быть не может. Сначала документы.",
  "Платить за воздух не собираюсь.",
  "Вы шутите? Документы покажите — потом обсудим.",
  "Не торопите события. Сначала проверим вашего работодателя и его бумаги.",
  "Нет. Сначала проверка, потом оплата после визы.",
  "Стоп. Документы → проверка → уже потом может будут деньги."
];

const SLOT_QUESTIONS = [
  "Как вы решаете проблему со слотами в посольство? Сколько ваши люди ждут?",
  "По регестрациям в посольство Чехии всё сложно. Как у вас с назначениями?",
  "Где записываете и в какие сроки? Конкретику дайте.",
  "Сколько времени уходит у ваших кандидатов на то чтобы получить слот? Реальные цифры."
];

const CRYPTO_SKEPTIC = [
  "Крипту не люблю. Рисково. Там одни машеники",
  "Крипта? Плохой знак. Лучше счёт-фактуру от фирмы, а не эти кошельки.",
  "Бывает, что берут криптой, но мне спокойнее через счёт-фактуру или банковским переводом.",
  "Если честно — криптой пользуются кто попало. Давайте нормальный инвойс."
];

const HR_CONTACT = [
  "Дайте контакт работодателя. Меня интересует реальный человек, а то с вами не возможно договорится.",
  "Нужен прямой контакт к работодателю. Без этого не двигаемся."
];

const BARGAIN = [
  "Дорого. Если дам сразу 10 кандидатов — будет скидка?",
  "А если я приведу пятёрку людей сразу, вы цену двинете?",
  "По цене давайте приземлимся. За одного столько не дам.",
  "Окей, но цифры не устраивают. Давайте обсуждать."
];

const URADPRACE_PUSH = [
  "Ссылку на uradprace дадите? Я проверю.",
  "Где проверка на uradprace? Без неё — слабая история.",
  "Покажите, где ваша позиция в реестре uradprace.",
  "У вас есть регистрация в реестре? Ссылку сюда."
];

const CANDIDATE_TEST = [
  "Хорошо, один кандидат на тест — возможно. Но сначала бумаги.",
  "Не спешим. Сначала проверка, потом дам одного человека в возможно.",
  "Один тестовый кандидат — обсуждаемо, но без бумаг не двигаюсь."
];

const CLOSING_READY = [
  "Окей. Давайте финализировать.",
  "Похоже, разобрались. Шлите инвойс — проверю и двинемся.",
  "Хорошо. Реквизиты присылайте, посмотрю и согласуем шаги."
];

// Запретные паттерны (грубый фильтр)
const BANNED_PATTERNS = [
  /кошел(е|ё)к|wallet/i,       // не даёт свои реквизиты
  /переведите.*мне/i,
  /я оплачу первым/i,
  /гарантирую.*виз/i,          // «гарантия визы»
  /связи.*посольств/i          // выдуманные связи
];

// ---------- Помощники ----------
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

// «оживляем» ответ: тон, вопросы, анти-ранняя оплата, анти-повторы
function humanize({ parsed, trust, evidences, history }) {
  let reply = String(parsed.reply || '').trim();
  const lastBot = lastAssistantReplyFromHistory(history);
  const uniqEvidenceCount = new Set(evidences || []).size;
  const gatesOk = (trust >= 90 && uniqEvidenceCount >= 2);

  // Запрещённые фразы → мягкий перехват к документам
  if (!reply || BANNED_PATTERNS.some(rx => rx.test(reply))) {
    reply = pick(ASK_DOCS_HUMAN);
    parsed.stage = parsed.stage === 'Payment' ? 'Contract' : (parsed.stage || 'Greeting');
    parsed.needEvidence = true;
    parsed.suggestedActions = ["ask_demands","ask_contract","ask_uradprace"];
    parsed.confidence = Math.min(parsed.confidence ?? trust, 70);
  }

  // Если модель «уходит» в оплату до ворот — жёсткий живой щелчок
  if (!gatesOk && parsed.stage === 'Payment') {
    reply = pick(TOO_EARLY_PAY_HUMAN);
    parsed.stage = 'Contract';
    parsed.needEvidence = true;
    const set = new Set(parsed.suggestedActions || []);
    set.add('ask_demands'); set.add('ask_contract'); set.add('ask_uradprace');
    parsed.suggestedActions = Array.from(set);
    parsed.confidence = Math.min(parsed.confidence ?? trust, 80);
  }

  // Поддержка «тем»: если в истории скоро после приветствия — прокидываем неудобные вопросы
  const userText = (history || []).filter(h=>h.role==='user').slice(-1)[0]?.content || '';
  if (/слот|запис|посольств/i.test(userText)) {
    reply = reply || pick(SLOT_QUESTIONS);
  }
  if (/крипт|crypto|usdt|btc/i.test(userText)) {
    reply = reply || pick(CRYPTO_SKEPTIC);
  }
  if (/hr|эйчар|кадр|отдел кадров|контакт/i.test(userText)) {
    reply = reply || pick(HR_CONTACT);
  }
  if (/сколько|цена|дорог|ценник|стоим/i.test(userText)) {
    reply = reply || pick(BARGAIN);
  }
  if (/uradprace|у?радпрац/i.test(userText)) {
    reply = reply || pick(URADPRACE_PUSH);
  }
  if (/кандидат|people|workers|людей/i.test(userText)) {
    reply = reply || pick(CANDIDATE_TEST);
  }

  // Ворота пройдены → короткое «финализируем»
  if (gatesOk && parsed.stage !== 'Payment') {
    const set = new Set(parsed.suggestedActions || []);
    set.add('invoice_request');
    parsed.suggestedActions = Array.from(set);
    reply = reply || pick(CLOSING_READY);
    parsed.stage = 'Payment';
    parsed.needEvidence = false;
    parsed.confidence = Math.max(parsed.confidence ?? 0, trust);
  }

  // Анти-повтор: если слово в слово как прошлый бот → другая фраза
  if (reply && lastBot && reply.trim().toLowerCase() === lastBot.trim().toLowerCase()) {
    reply = pick(FALLBACK_HUMAN);
  }

  parsed.reply = trimToSentences(reply, 6);
  if (!parsed.reply) parsed.reply = pick(FALLBACK_HUMAN);

  return parsed;
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
    temperature: 0.35,
    top_p: 0.9,
    frequency_penalty: 0.3,
    presence_penalty: 0.0,
    max_tokens: 380,
    response_format: { type: 'json_object' },
    messages
  });

  // --- извлекаем валидный JSON
  const raw = resp.choices?.[0]?.message?.content || '{}';
  const json = extractFirstJsonObject(raw);
  let parsed = json ?? {
    reply: pick(FALLBACK_HUMAN),
    confidence: clamp(trust, 0, 40),
    stage: stage || "Greeting",
    needEvidence: true,
    suggestedActions: ["ask_demands","ask_contract","ask_uradprace"]
  };

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

// /api/reply — основной
app.post('/api/reply', async (req, res) => {
  try {
    const b = req.body || {};

    // Нормализация evidences
    const evidences =
      Array.isArray(b.evidences) ? b.evidences.map(String) :
      (Number.isFinite(b.evidence) ? Array.from({ length: Math.max(0, b.evidence|0) }, (_, i) => `proof_${i+1}`) :
      []);

    // История
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
      meta: {
        ok: true,
        trust,
        evidenceCount,
        stage: result.stage,
        actions: result.suggestedActions
      }
    });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// /api/score — подсказки менеджеру (лёгкая эвристика)
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
    if (/renovogo|renovogo\.com/i.test(msgText)) good.push('Дали проверяемый факт');
    if (evidences.length >= 2) good.push('Приложили ≥2 доказательства'); else bad.push('Мало доказательств');
    if (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText)) good.push('Есть финальный CTA');

    const final = clamp(Math.round(
      (/(здрав|прив|добрый)/i.test(msgText) ? 15 : 0) +
      (/renovogo|renovogo\.com/i.test(msgText) ? 15 : 0) +
      ((evidences.length >= 2) ? 35 : 0) +
      (/(контракт|сч[её]т|инвойс|готовы начать)/i.test(msgText) ? 35 : 0)
    ), 0, 100);

    res.json({ final, good, bad, trust, evidences: evidences.length });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

// Совместимость со старым роутом
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
