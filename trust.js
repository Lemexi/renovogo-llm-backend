// trust.js
// Доверие Али: правило-ориентированный скоринг с «человечной» кривой.
// Совместимо с server.js v2025-09-16-5 (ключи evidences уже нормализованы).

// ==== Категории доказательств (нормализованные ключи) ====
// HARD: документы-основания для работы.
const HARD_PROOFS = new Set([
  'demand_letter',      // официальный деманд/запрос (без реквизитов)
  'coop_contract_pdf',  // ПОЛНЫЙ контракт о сотрудничестве (подпись/печать, реквизиты)
  'contract_pdf'        // алиас со старого фронта (тоже трактуем как полный)
]);

// MEDIUM: проверяемые вторичные признаки.
const MEDIUM_PROOFS = new Set([
  'website',            // сайт
  'company_registry',   // выписка из реестра компаний
  'registry_proof',     // подтверждение регистрации/уряд праце
  'visa_sample',        // пример визы/штампа (деперсон.)
  'reviews'             // отзывы
]);

// SUPPORT: вспомогательные материалы.
const SUPPORT_PROOFS = new Set([
  'sample_contract_pdf',// ПРИМЕР контракта (не финал)
  'price_breakdown',    // смета/состав пакета
  'slot_plan',          // план по слотам
  'nda',
  'invoice_template',
  'business_card',      // визитка менеджера
  'presentation',
  'video'
]);

// Подсчёт типов пруфов (учитываем уникальные ключи)
function countKinds(evidences = []) {
  const uniqSet = new Set((evidences || []).map(e => String(e).trim().toLowerCase()));
  let hard = 0, med = 0, sup = 0;
  for (const e of uniqSet) {
    if (HARD_PROOFS.has(e)) hard++;
    else if (MEDIUM_PROOFS.has(e)) med++;
    else if (SUPPORT_PROOFS.has(e)) sup++;
  }
  return { uniq: uniqSet.size, hard, med, sup, uniqSet };
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

// Нереалистично быстрые сроки для доков/визы (<=5 рабочих дней)
function unrealisticTimeline(text='') {
  const t = String(text).toLowerCase();
  const talksDocs = /(документ|контракт|офер|виза|слот|приглашени|регистрац)/.test(t);
  const fast = /\b(1|2|3|4|5)\s*(дн(я|ей)?|day|days|сут|час(а|ов)?|hour|hours)\b|48\s*час|завтра/.test(t);
  return talksDocs && fast;
}

// Сводный тон по последним репликам
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

// Сигналы/красные флаги из последней фразы пользователя
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

  // НОВОЕ: путаница «€350 = зарплата» (это сервисный платёж)
  if (/(зарплат|salary)/.test(t) && /(€\s*350|350\s*€|\b350\b)/.test(t)) red.push('fee_salary_confusion');

  // НОВОЕ: «реквизиты из Demand» — концептуальная ошибка
  if (/реквизит/.test(t) && /demand/.test(t)) red.push('requisites_from_demand');

  // GREEN HINTS (мягкие позитивные сигналы)
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
  let score = clamp01_100(baseTrust);

  // ===== 1) Документальные признаки (перевзвешенные) =====
  // HARD = +8 за каждый, MEDIUM = +4, SUPPORT = +2
  // Лёгкие анти-накрутки: считаем MED до 3шт, SUPPORT до 3шт
  const { uniq, hard, med, sup, uniqSet } = countKinds(evidences);
  const hasCoop = uniqSet.has('coop_contract_pdf') || uniqSet.has('contract_pdf');

  score += hard * 10;
  score += Math.min(3, med) * 4;
  score += Math.min(3, sup) * 2;
  if (hasCoop) score += 4; // полный контракт всё ещё даёт доп. доверие

  // Разнообразие типов (стимулируем нести разные пруфы)
  if (uniq >= 3) score += 3;
  if (uniq >= 5) score += 3;

  // ===== 2) Стадии (мягкие бонусы, если есть реальные доки) =====
  const stages = (history || []).map(h => String(h.stage || '').toLowerCase());
  const sawContractStage  = stages.includes('contract');
  const sawCandidateStage = stages.includes('candidate');
  const sawPaymentStage   = stages.includes('payment');

  if (sawCandidateStage && (hard + med) > 0) score += 2;
  if (sawContractStage  && hard > 0)         score += 3;

  // ===== 3) Тональность и сигналы последней фразы =====
  const tone = toneFromHistory(history);
  score += Math.min(4, tone.polite); // небольшой плюс за вежливость
  score += tone.press;               // минусы за давление
  if (tone.obseq >= 2) score -= 3;   // чрезмерная угодливость снижает доверие слегка

  const sig = textSignals(lastUserText || '');
  for (const r of sig.red) {
    if (r === 'crypto_upfront')                 score -= 25;
    else if (r === 'impossible_guarantee')      score -= 30;
    else if (r === 'pressure')                  score -= 16; 
    else if (r === 'embassy_connections_claim') score -= 30;
    else if (r === 'unrealistic_timeline')      score -= 12;
    else if (r === 'fee_salary_confusion')      score -= 12;
    else if (r === 'requisites_from_demand')    score -= 10;
  }
  if (sig.green.includes('mentions_bank_payment')) score += 2;
  if (sig.green.includes('mentions_website'))      score += 1;
  if (sig.green.includes('mentions_demand'))       score += 2;
  if (sig.green.includes('mentions_contract'))     score += 2;
  if (sig.green.includes('test_one_candidate'))    score += 2;

  // ===== 4) Микро-кредиты за «нормальный» диалог (без давления) =====
  const micro = dialogMicroCredits(history);
  score += Math.min(8, micro.uniqInfoKeysCount);
  if (!micro.recentEarlyPayPressure) score += 1;

  // ===== 5) Нелинейные ворота (пересчитаны под новые веса) =====
  if (score > 30 && hard < 1) score = 30; // Gate 1
  const passGate2 = (hard >= 2) || (hard >= 1 && med >= 1);
  if (score > 60 && !passGate2) score = 60; // Gate 2
  if (score > 75 && (hard < 2 || sig.red.length > 0)) score = 75; // Gate 3
  if (sawPaymentStage && score < 75) score = Math.max(0, score - 8);

  // Финал
  return Math.round(clamp01_100(score));

  // ===== Вспомогательные =====
  function clamp01_100(x){ return Math.max(0, Math.min(100, Number(x) || 0)); }

  function dialogMicroCredits(hist) {
    const userMsgs = (hist || []).filter(m => String(m.role||'').toLowerCase()==='user').slice(-12);

    const KEY_REGEXPS = {
      name: /(как.*зовут|вас зовут|имя|name)/i,
      office: /(офис|office|head\s*office|штаб|hq)/i,
      location: /(где.*(наход|располож)|город|city|address|адрес)/i,
      years: /(сколько.*лет|на рынке|лет в бизнесе|since\s+\d{4}|основан(ы|а)|founded)/i,
      specialization: /(чем занимаетесь|специализац|на чём фокус|что делаете|services|услуги)/i,
      registration: /(регистрац|ico|ičo|krs|regon|edrpou|uic|inn|nip|ico|номер компании|company number)/i,
      team: /(штат|сотрудник|сколько человек|команда|headcount|staff)/i
    };

    const foundKeys = new Set();
    for (const m of userMsgs) {
      const t = (m.text || m.content || m.message || '').toString();
      for (const [key,rx] of Object.entries(KEY_REGEXPS)) {
        if (rx.test(t)) foundKeys.add(key);
      }
    }

    const payRx = /(оплат|плат[её]ж|инвойс|сч[её]т| реквизит(ы)?)/i;
    const earlyPayHits = userMsgs.slice(-6).filter(m => payRx.test((m.text||m.content||'')+'')).length;
    const recentEarlyPayPressure = (earlyPayHits >= 2) && !passGate2;

    return {
      uniqInfoKeysCount: foundKeys.size,
      recentEarlyPayPressure
    };
  }
}
