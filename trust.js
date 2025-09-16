// trust.js
// Жёсткий, пороговый скоринг доверия Али.

const HARD_PROOFS = new Set([
  'demand_letter',    // официальный деманд/запрос по вакансии
  'contract_pdf'      // контракт/офер/соглашение о сотрудничестве (PDF)
]);

const MEDIUM_PROOFS = new Set([
  'website',          // сайт компании/работодателя
  'company_registry', // выписка из гос.реестра компаний
  'visa_sample'       // пример визы/штампа (деперсонализированный)
]);

const SUPPORT_PROOFS = new Set([
  'price_breakdown',  // прозрачная смета/что входит в пакет
  'slot_plan',        // план по слотам/термины (если уместно)
  'nda',
  'invoice_template',
  'business_card',
  'presentation'
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

  // GREEN HINTS (мягкие)
  if (/счёт|инвойс|банковск/.test(t)) green.push('mentions_bank_payment');
  if (/сайт|website|https?:\/\//.test(t)) green.push('mentions_website');
  if (/деманд|demand/.test(t)) green.push('mentions_demand');
  if (/контракт|офер|соглашени/i.test(t)) green.push('mentions_contract');
  if (/тест(овый)? кандидат|с одного кандидата/.test(t)) green.push('test_one_candidate');

  // GRAY
  if (/позже|вернусь|через неделю|давайте потом/.test(t)) gray.push('postpone');

  return { red, green, gray };
}

export function computeTrust({ baseTrust = 20, evidences = [], history = [], lastUserText = '' }) {
  // База
  let score = Math.max(0, Math.min(100, baseTrust));

  // Доказательства по категориям
  const { uniq, hard, med, sup } = countKinds(evidences);

  // Весовые добавки
  score += hard * 18;
  score += Math.min(2, med) * 8;
  score += Math.min(2, sup) * 3;

  // Бонус за разнообразие типов
  if (uniq >= 3) score += 6;
  if (uniq >= 5) score += 6;

  // Стадийные мягкие бонусы (если есть что-то реальное)
  const stages = (history || []).map(h => (h.stage || '').toLowerCase());
  const sawContractStage = stages.includes('contract');
  const sawCandidateStage = stages.includes('candidate');
  const sawPaymentStage   = stages.includes('payment');

  if (sawCandidateStage && (hard + med) > 0) score += 4;
  if (sawContractStage  && hard > 0) score += 5;

  // Текстовые сигналы
  const sig = textSignals(lastUserText || '');
  for (const r of sig.red) {
    if (r === 'crypto_upfront') score -= 25;
    else if (r === 'impossible_guarantee') score -= 30;
    else if (r === 'pressure') score -= 20;
    else if (r === 'embassy_connections_claim') score -= 30;
  }
  if (sig.green.includes('mentions_bank_payment')) score += 3;
  if (sig.green.includes('mentions_website')) score += 2;
  if (sig.green.includes('mentions_demand')) score += 3;
  if (sig.green.includes('mentions_contract')) score += 3;
  if (sig.green.includes('test_one_candidate')) score += 3;

  // Нелинейные ворота (без «реестров»/HR)
  // Gate 1: >50 требует минимум 1 тяжёлый пруф
  if (score > 50 && hard < 1) score = 50;
  // Gate 2: >80 требует (2 тяжёлых) ИЛИ (1 тяжёлый + 1 средний)
  const passGate2 = (hard >= 2) || (hard >= 1 && med >= 1);
  if (score > 80 && !passGate2) score = 80;
  // Gate 3: >90 требует 2 тяжёлых и отсутствие красных флагов
  if (score > 90 && (hard < 2 || sig.red.length > 0)) score = 90;

  // Наказание за ранний «Payment», если ворота не пройдены
  if (sawPaymentStage && score < 90) score = Math.max(0, score - 10);

  // Финальная нормализация
  score = Math.round(Math.max(0, Math.min(100, score)));
  return score;
}
