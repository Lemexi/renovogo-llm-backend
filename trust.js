// trust.js
// Жёсткий, пороговый скоринг доверия Али.
// computeTrust({ baseTrust, evidences, history, lastUserText }) -> number (0..100)

const HARD_PROOFS = new Set([
  'demand_letter',          // официальный деманд
  'contract_pdf',           // подписной контракт/офер
  'uradprace_link',         // ссылка/скрин с uradprace.cz (или иной гос.реестр)
  'employer_contact'        // прямой рабочий контакт HR/работодателя (email/phone на домене)
]);

const MEDIUM_PROOFS = new Set([
  'website',                // сайт компании/работодателя
  'company_registry',       // выписка из гос.реестра компаний
  'visa_sample',            // пример визы/штампа (деперсонализированный)
  'reviews', 'case_reviews' // кейсы/отзывы с верифицируемыми деталями
]);

const SUPPORT_PROOFS = new Set([
  'price_breakdown',        // прозрачная смета/что входит в пакет
  'slot_plan',              // чёткий план по слотам/термины/каналы
  'nda', 'invoice_template','business_card','presentation'
]);

function countKinds(evidences = []) {
  const uniq = new Set(evidences.map(e => String(e).trim().toLowerCase()));
  let hard = 0, med = 0, sup = 0;
  for (const e of uniq) {
    if (HARD_PROOFS.has(e)) hard++;
    else if (MEDIUM_PROOFS.has(e)) med++;
    else if (SUPPORT_PROOFS.has(e)) sup++;
  }
  return { uniq: uniq.size, hard, med, sup };
}

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

  // GREEN HINTS (мягкие признаки)
  if (/uradprace|mzv|gov/i.test(text)) green.push('mentions_official_registry');
  if (/счёт|инвойс|банковск/.test(t)) green.push('mentions_bank_payment');
  if (/сайт|website|https?:\/\//.test(t)) green.push('mentions_website');
  if (/деманд|demand/.test(t)) green.push('mentions_demand');
  if (/контракт|офер/.test(t)) green.push('mentions_contract');
  if (/тест(овый)? кандидат|с одного кандидата/.test(t)) green.push('test_one_candidate');

  // GRAY (неопределённые)
  if (/позже|вернусь|через неделю|давайте потом/.test(t)) gray.push('postpone');

  return { red, green, gray };
}

export function computeTrust({ baseTrust = 20, evidences = [], history = [], lastUserText = '' }) {
  // База
  let score = Math.max(0, Math.min(100, baseTrust));

  // Доказательства по категориям
  const { uniq, hard, med, sup } = countKinds(evidences);

  // Весовые добавки (жёсткие)
  // — Сначала накидываем за "тяжёлые"
  score += hard * 18;            // каждая тяжёлая — мощный буст
  score += Math.min(2, med) * 8; // максимум 2 средние по 8
  score += Math.min(2, sup) * 3; // максимум 2 поддерживающие по 3

  // Бонус за разнообразие типов
  if (uniq >= 3) score += 6;
  if (uniq >= 5) score += 6;

  // Стадийные мягкие бонусы (только если есть хоть что-то реальное)
  const stages = (history || []).map(h => (h.stage || '').toLowerCase());
  const sawContractStage = stages.includes('contract');
  const sawCandidateStage = stages.includes('candidate');
  const sawPaymentStage   = stages.includes('payment');

  if (sawCandidateStage && (hard + med) > 0) score += 4;
  if (sawContractStage  && hard > 0) score += 5;

  // Текстовые сигналы
  const sig = textSignals(lastUserText || '');
  // Красные флаги — сильные штрафы
  for (const r of sig.red) {
    if (r === 'crypto_upfront') score -= 25;
    else if (r === 'impossible_guarantee') score -= 30;
    else if (r === 'pressure') score -= 20;
    else if (r === 'embassy_connections_claim') score -= 30;
  }
  // Зелёные — небольшие плюсы (без документов — не спасут)
  if (sig.green.includes('mentions_official_registry')) score += 4;
  if (sig.green.includes('mentions_bank_payment')) score += 3;
  if (sig.green.includes('mentions_website')) score += 2;
  if (sig.green.includes('mentions_demand')) score += 3;
  if (sig.green.includes('mentions_contract')) score += 3;
  if (sig.green.includes('test_one_candidate')) score += 3;

  // Нелинейные "ворота" — без тяжёлых доказательств потолок
  // Ворота 1: чтобы перелезть за 50, нужен минимум 1 тяжёлый пруф
  if (score > 50 && hard < 1) score = 50;

  // Ворота 2: чтобы перелезть за 80, нужно минимум 2 тяжёлых ИЛИ 1 тяжёлый + 1 средний + упоминание оф.реестра
  const hasRegistryMention = sig.green.includes('mentions_official_registry');
  const passGate2 = (hard >= 2) || (hard >= 1 && med >= 1 && hasRegistryMention);
  if (score > 80 && !passGate2) score = 80;

  // Ворота 3 (финал): чтобы попасть к 90+, нужно:
  //  - минимум 2 тяжёлых
  //  - И отсутствие красных флагов
  if (score > 90 && (hard < 2 || sig.red.length > 0)) score = 90;

  // Наказание за ранний «Payment», если ворота не пройдены
  if (sawPaymentStage && score < 90) score = Math.max(0, score - 10);

  // Финальная нормализация
  score = Math.round(Math.max(0, Math.min(100, score)));
  return score;
}
