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

// CORS
const allowed = (process.env.ALLOWED_ORIGINS || '').split(',').map(s => s.trim()).filter(Boolean);
app.use(cors({
  origin: (origin, cb) => {
    if (!origin) return cb(null, true);
    if (allowed.length === 0 || allowed.includes(origin)) return cb(null, true);
    cb(new Error('Not allowed by CORS'));
  },
  credentials: false
}));

// Groq client
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const MODEL = process.env.GROQ_MODEL || 'llama-3.1-8b-instant';

// Валидация входа
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

// Хелпер: формируем сообщения для LLM
function buildMessages({ history = [], message, trust, evidences }) {
  const sys = { role: 'system', content: SYSTEM_PROMPT + `\nТекущий trust=${trust}. Доказательства=${JSON.stringify(evidences||[])}.` };

  // Контекстные реплики из истории (ужатые)
  const trimmed = history.slice(-10).map(h => ({
    role: h.role,
    content: h.content
  }));

  return [
    sys,
    ...trimmed,
    { role: 'user', content: message }
  ];
}

// Основной эндпоинт
app.post('/chat', async (req, res) => {
  try {
    const data = ChatSchema.parse(req.body);
    const trust = computeTrust({
      baseTrust: 30,
      evidences: data.evidences || [],
      history: (data.history || []).filter(h => h.stage).map(h => ({ stage: h.stage }))
    });

    const messages = buildMessages({ history: data.history, message: data.message, trust, evidences: data.evidences });

    const resp = await groq.chat.completions.create({
      model: MODEL,
      temperature: 0.5,
      response_format: { type: 'json_object' }, // просим строго JSON
      messages
    });

    const raw = resp.choices?.[0]?.message?.content || '{}';
    let parsed;
    try { parsed = JSON.parse(raw); } catch {
      parsed = { reply: "Не понял. Давайте вернёмся к разговору.", confidence: 40, stage: "Greeting", needEvidence: true, suggestedActions: [] };
    }

    // Принудительная логика финализации:
    const evidenceCount = new Set(data.evidences || []).size;
    if (trust >= 90 && evidenceCount >= 2) {
      if (!Array.isArray(parsed.suggestedActions) || !parsed.suggestedActions.includes('invoice_request')) {
        parsed.suggestedActions = [...(parsed.suggestedActions||[]), 'invoice_request'];
        parsed.reply = parsed.reply || "Финализируем: пришлите реквизиты/кошелёк.";
        parsed.stage = 'Payment';
      }
    } else {
      // Если доверия/доков не хватает — не позволяем перейти к оплате
      if (parsed.stage === 'Payment' && !(trust >= 90 && evidenceCount >= 2)) {
        parsed.stage = 'Contract';
        parsed.reply = "Пока рано к оплате. Покажите ещё документы (контракт/пример визы/деманд).";
      }
    }

    res.json({
      ok: true,
      trust,
      evidenceCount,
      result: parsed
    });
  } catch (e) {
    console.error(e);
    res.status(400).json({ ok: false, error: String(e.message || e) });
  }
});

app.get('/health', (_, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`LLM backend running on :${PORT}`));