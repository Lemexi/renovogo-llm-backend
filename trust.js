// trust.js
// Доверие Али: правило-ориентированный скоринг с «человечной» кривой.
// Совместимо с server.js v2025-09-16-2 (ключи evidences уже нормализованы).

// ==== Категории доказательств (нормализованные ключи) ====
// HARD: документы, на основании которых реально можно работать.
const HARD_PROOFS = new Set([
  'demand_letter',      // официальный деманд/запрос
  'coop_contract_pdf',  // ПОЛНЫЙ контракт о сотрудничестве (подпись/печать)
  'contract_pdf'        // алиас, если прилетит со старого фронта
]);

// MEDIUM: проверяемые, но вторичные признаки.
const MEDIUM_PROOFS = new Set([
  'website',            // сайт
  'company_registry',   // выписка из реестра компаний (если прилетит таким ключом)
  'registry_proof',     // подтверждение регистрации/уряд праце и т.п.
  'visa_sample',        // пример визы/штампа (деперсон.)
  'reviews'             // отзывы (если есть регулярный ключ)
]);

// SUPPORT: вспомогательные материалы.
const SUPPORT_PROOFS = new Set([
  'sample_contract_pdf',// ПРИМЕР контракта (не финал)
  'price_breakdown',    // смета/состав пакета
  'slot_plan',          // план по слотам
  'nda',
  'invoice_template',
  'business_card',
  'presentation',
  'video'
]);

// Подсчёт типов пруфов
function countKinds(evidences = []) {
  const uniq = new Set((evidences || []).map(e => String(e).trim().toLowerCase()));
  let hard = 0, med = 0, sup = 0;
  for (const e of uniq) {
    if (HARD_PROOFS.has(e)) hard++;
    else if (MEDIUM_PROOFS.has(e)) med++;
    else if (SUPPORT_PROOFS.has(e)) sup++;
  }
  return { uniq: uniq.size, hard, med, sup };
}

// ==== Анализ тона и сигналов ====

function politenessScore(text='') {
  const t = String(text).toLowerCase();
  let sc = 0;
  if (/(здрав|добрый день|доброе утро|добрый вечер|приветствую)/.test(t)) sc += 2;
  if (/(спасибо|благодарю|пожалуйста)/.test(t)) sc += 1;
  if (/(рад(а)? знакомству|приятно познакомиться)/.test(t)) sc += 2;
  return sc;
}

function pressureScore(text='') {
  const t = String(text).toLowerCase();
  let sc = 0;
  if (/(срочно|немедленно|давайте быстрее|прямо сейчас|сегодня же)/.test(t)) sc -= 4;
  if (/(или мы уйд[её]м|иначе|последний шанс)/.test(t)) sc -= 4;
  return sc;
}

// Нереалистично быстрые сроки для документов/визы (<=5 рабочих дней)
function unrealisticTimeline(text='') {
  const t = String(text).toLowerCase();
  const talksDocs = /(документ|контракт|офер|виза|слот|приглашени|регистрац)/.test(t);
  const fast = /\b(1|2|3|4|5)\s*(дн(я|ей)?|day|days|сут|час(а|ов)?|hour|hours)\b|48\s*час|завтра/.test(t);
  return talksDocs && fast;
}

// Сводный тон последних сообщений истории
function toneFromHistory(history=[]) {
  const lastMsgs = (history || []).slice(-6);
  let polite = 0, press = 0, obseq = 0;

  for (const h of lastMsgs) {
    if (!h || !h.content) continue;
    const c = String(h.content);
    polite += politenessScore(c);
    press  += pressureScore(c);
    // Слишком быстрое соглашательство без фактов
    if (/(ок|окей|да, конечно|что угодно|как скажете)/i.test(c) && /подбор|ищите|занимайтесь/i.test(c)) {
      obseq += 1;
    }
  }
  return { polite, press, obseq };
}

// Сигналы из последней фразы пользователя
function textSignals(text = '') {
  const t = (text || '').toLowerCase();

  const red = [];
  const green = [];
  const gray = [];

  // RED FLAGS
  if (/\bкрипт|usdt|btc|eth\b/.test(t) && /сразу|предоплат|аванс/.test(t)) red.push('crypto_upfront');
  if (/гарантирую|100%|сто процентов|без отказов/.test(t)) red.push('impossible_guarantee');
  if (/поторопитесь|только сегодня|срочно платите/.test(t)) red.push('pressure');
  if (/связи в посольстве|решаем через знакомых/.test(t)) red.push('embassy_connections_claim');
  if (unrealisticTimeline(t)) red.push('unrealistic_timeline');

  // GREEN HINTS (мягкие)
  if (/сч[её]т|инвойс|банковск/.test(t)) green.push('mentions_bank_payment');
  if (/сайт|website|https?:\/\//.test(t)) green.push('mentions_website');
  if (/деманд|demand/.test(t)) green.push('mentions_demand');
  if (/контракт|офер|соглашени/i.test(t)) green.push('mentions_contract');
  if (/тест(овый)? кандидат|с одного кандидата|1-?2 кандидата/i.test(t)) green.push('test_one_candidate');

  // GRAY
  if (/позже|вернусь|через неделю|давайте потом/.test(t)) gray.push('postpone');

  return { red, green, gray };
}

// ==== Основной расчёт ====
export function computeTrust({ baseTrust = 20, evidences = [], history = [], lastUserText = '' }) {
  // База
  let score = Math.max(0, Math.min(100, baseTrust));

  // Документальные признаки
  const { uniq, hard, med, sup } = countKinds(evidences);
  score += hard * 18;                 // каждый тяжёлый — сильный прирост
  score += Math.min(2, med) * 8;      // максимум 16 за медиум
  score += Math.min(2, sup) * 3;      // максимум 6 за саппорт

  // Разнообразие типов
  if (uniq >= 3) score += 6;
  if (uniq >= 5) score += 6;

  // Стадийные мягкие бонусы (если есть что-то реальное)
  const stages = (history || []).map(h => (h.stage || '').toLowerCase());
  const sawContractStage  = stages.includes('contract');
  const sawCandidateStage = stages.includes('candidate');
  const sawPaymentStage   = stages.includes('payment');

  if (sawCandidateStage && (hard + med) > 0) score += 4;
  if (sawContractStage  && hard > 0) score += 5;

  // Тон последних сообщений
  const tone = toneFromHistory(history);
  score += Math.min(4, tone.polite);  // небольшой плюс за вежливость
  score += tone.press;                // минусы за давление
  if (tone.obseq >= 2) score -= 4;    // чрезмерное «ок, как скажете»

  // Сигналы из последней фразы пользователя
  const sig = textSignals(lastUserText || '');
  for (const r of sig.red) {
    if (r === 'crypto_upfront')                 score -= 25;
    else if (r === 'impossible_guarantee')      score -= 30;
    else if (r === 'pressure')                  score -= 18;
    else if (r === 'embassy_connections_claim') score -= 30;
    else if (r === 'unrealistic_timeline')      score -= 12;
  }
  if (sig.green.includes('mentions_bank_payment')) score += 3;
  if (sig.green.includes('mentions_website'))      score += 2;
  if (sig.green.includes('mentions_demand'))       score += 3;
  if (sig.green.includes('mentions_contract'))     score += 3;
  if (sig.green.includes('test_one_candidate'))    score += 3;

  // Нелинейные ворота (чтобы нельзя было «накрутить» мелочами)
  // Gate 1: >50 требует минимум 1 тяжёлый пруф
  if (score > 50 && hard < 1) score = 50;

  // Gate 2: >80 требует (2 тяжёлых) ИЛИ (1 тяжёлый + 1 средний)
  const passGate2 = (hard >= 2) || (hard >= 1 && med >= 1);
  if (score > 80 && !passGate2) score = 80;

  // Gate 3: >90 требует 2 тяжёлых и отсутствие красных флагов
  if (score > 90 && (hard < 2 || sig.red.length > 0)) score = 90;

  // Наказание за ранний «Payment», если ворота не пройдены
  if (sawPaymentStage && score < 90) score = Math.max(0, score - 10);

  // Финал
  score = Math.round(Math.max(0, Math.min(100, score)));
  return score;
}
