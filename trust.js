// trust.js
// Доверие Али: правило-ориентированный скоринг с «человечной» кривой.
// Совместимо с server.js v2025-09-16-11 (ключи evidences уже нормализованы).

/*
  НОВОЕ:
  — Персональные вопросы (про семью/личное):
      Порог/лимиты (глобальный лимит на всю беседу, не «обнуляется»):
        40+  → максимум 2 вопроса,    бонус +1
        50+  → максимум 3 вопроса,    бонус +1
        60+  → максимум 4 вопроса,    бонус +1
        70+  → максимум 5 вопросов,   бонус +2
        80+  → максимум 6 вопросов,   бонус +2
        90+  → максимум 7 вопросов,   бонус +3
        100+ → максимум 8 вопросов,   бонус +3
      Правило «квота не суммируется»: если на 50 уже спросили 3 личных, то на 70 останется 2 из 5.
      Реализация: считаем ОБЩЕЕ количество личных вопросов пользователя в истории и сравниваем с лимитом
                  по текущему trust-бракету; сверх лимита — бонус не даём (и штрафов тоже не даём).
                  На раннем этапе (<40) за личные вопросы — небольшой минус (−4).
  — Вежливость: за «сэр/пожалуйста/извините/подскажите» давать +0.5, но не спамить.
      Кулдаун ~360 сек: т.к. таймштампов нет, используем эквивалент 6 ходов (COOLDOWN_TURNS = 6).
*/

const COOLDOWN_TURNS = 6; // ~эквивалент 360с без таймштампов

// ==== Категории доказательств (нормализованные ключи) ====
const HARD_PROOFS = new Set([
  'demand_letter',
  'coop_contract_pdf',
  'contract_pdf'
]);

const MEDIUM_PROOFS = new Set([
  'website',
  'company_registry',
  'registry_proof',
  'visa_sample',
  'reviews'
]);

const SUPPORT_PROOFS = new Set([
  'sample_contract_pdf',
  'price_breakdown',
  'slot_plan',
  'nda',
  'invoice_template',
  'business_card',
  'presentation',
  'video'
]);

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

// ==== Анализ тона и сигналы ====

function politenessScoreLite(text='') {
  const t = String(text).toLowerCase();
  let sc = 0;
  if (/(здрав|добрый день|доброе утро|добрый вечер|приветствую)/.test(t)) sc += 2;
  if (/(спасибо|благодарю|пожалуйста)/.test(t)) sc += 1;
  if (/(рад(а)? знакомству|приятно познакомиться)/.test(t)) sc += 2;
  if (/renovogo(\.com)?/i.test(t)) sc += 1;
  return sc;
}

function pressureScore(text='') {
  const t = String(text).toLowerCase();
  let sc = 0;
  if (/(срочно|немедленно|давайте быстрее|прямо сейчас|сегодня же)/.test(t)) sc -= 4;
  if (/(или мы уйд[её]м|иначе|последний шанс)/.test(t)) sc -= 4;
  return sc;
}

function unrealisticTimeline(text='') {
  const t = String(text).toLowerCase();
  const talksDocs = /(документ|контракт|офер|виза|слот|приглашени|регистрац)/.test(t);
  const fast = /\b(1|2|3|4|5)\s*(дн(я|ей)?|day|days|сут|час(а|ов)?|hour|hours)\b|48\s*час|завтра/.test(t);
  return talksDocs && fast;
}

// Конкретика: числа/даты/локации/валюты
function concretenessScore(text='') {
  const t = String(text);
  let sc = 0;
  if (/\b\d{1,3}(\s?[-–]\s?\d{1,3})?\b/.test(t)) sc += 1;
  if (/(€|\bEUR\b|\bCZK\b|\bPLN\b|\$|\bUSD\b)/i.test(t)) sc += 1;
  if (/\b(янв|фев|мар|апр|май|июн|июл|авг|сен|окт|ноя|дек|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*/i.test(t) ||
      /\b20\d{2}\b/.test(t)) sc += 1;
  if (/\b(Прага|Brno|Пльзень|Warsaw|Варшава|Wrocław|Вроцлав|Katowice|Катовице|Чех|Польш)\w*/i.test(t)) sc += 1;
  return Math.min(sc, 3);
}

function toneFromHistory(history=[]) {
  const lastMsgs = (history || []).slice(-6);
  let polite = 0, press = 0, obseq = 0;
  for (const h of lastMsgs) {
    if (!h || !h.content) continue;
    const c = String(h.content);
    polite += politenessScoreLite(c);
    press  += pressureScore(c);
    if (/(ок($|[.!?])|окей|да, конечно|что угодно|как скажете)/i.test(c) && /подбор|ищите|занимайтесь/i.test(c)) {
      obseq += 1;
    }
  }
  return { polite, press, obseq };
}

function textSignals(text = '') {
  const t = (text || '').toLowerCase();
  const red = [], green = [], gray = [];

  if (/\bкрипт|usdt|btc|eth\b/.test(t) && /сразу|предоплат|аванс/.test(t)) red.push('crypto_upfront');
  if (/гарантирую|100%|сто процентов|без отказов/.test(t)) red.push('impossible_guarantee');
  if (/поторопитесь|только сегодня|срочно платите/.test(t)) red.push('pressure');
  if (/связи в посольстве|решаем через знакомых/.test(t)) red.push('embassy_connections_claim');
  if (unrealisticTimeline(t)) red.push('unrealistic_timeline');

  if (/(зарплат|salary)/.test(t) && /(€\s*350|350\s*€|\b350\b)/.test(t)) red.push('fee_salary_confusion');
  if (/реквизит/.test(t) && /demand/.test(t)) red.push('requisites_from_demand');

  if (/сч[её]т|инвойс|банковск/.test(t)) green.push('mentions_bank_payment');
  if (/сайт|website|https?:\/\//.test(t)) green.push('mentions_website');
  if (/деманд|demand/.test(t)) green.push('mentions_demand');
  if (/контракт|офер|соглашени/i.test(t)) green.push('mentions_contract');
  if (/тест(овый)? кандидат|с одного кандидата|1-?2 кандидата/i.test(t)) green.push('test_one_candidate');

  if (/позже|вернусь|через неделю|давайте потом/.test(t)) gray.push('postpone');

  return { red, green, gray };
}

function businessFocusScore(text='') {
  const t = String(text).toLowerCase();
  let sc = 0;
  if (/(ваканси|позици|role|position|должност)/.test(t)) sc += 1;
  if (/(зарплат|salary|нетто|брутто)/.test(t)) sc += 1;
  if (/(жиль|accommodat|общежит|проживан)/.test(t)) sc += 1;
  if (/(график|смен|hours|schedule|work\s*time)/.test(t)) sc += 1;
  if (/(локац|город|location|where)/.test(t)) sc += 1;
  if (/(контракт|demand|документ|офер|agreement)/.test(t)) sc += 1;
  return Math.min(3, sc);
}

// ==== ЛИЧНЫЕ ВОПРОСЫ ====

const PERSONAL_RX = /(семья|дети|женат|замужем|личн(ая|ые)|хобби|возраст|год(а)?|сколько лет|любим(ый|ая)|увлечен|чем увлекаешь|откуда ты|где живешь|семейное положение)/i;

function countPersonalQuestions(history=[]) {
  // считаем количество сообщений пользователя, где есть "личная" тема (по 1 за сообщение)
  const userMsgs = (history || []).filter(m => String(m.role||'').toLowerCase()==='user');
  let cnt = 0;
  for (const m of userMsgs) {
    const t = (m.text || m.content || m.message || '').toString();
    if (PERSONAL_RX.test(t)) cnt++;
  }
  return cnt;
}

function personalQuestionsCapByTrust(trust=0) {
  if (trust >= 100) return { cap: 8, bonus: 3 };
  if (trust >= 90)  return { cap: 7, bonus: 3 };
  if (trust >= 80)  return { cap: 6, bonus: 2 };
  if (trust >= 70)  return { cap: 5, bonus: 2 };
  if (trust >= 60)  return { cap: 4, bonus: 1 };
  if (trust >= 50)  return { cap: 3, bonus: 1 };
  if (trust >= 40)  return { cap: 2, bonus: 1 };
  return { cap: 0, bonus: 0 }; // <40 — не поощряем, будет минус
}

function personalLifeAdjustment({ trustNow=0, history=[], lastUserText='' }) {
  const askedPersonalNow = PERSONAL_RX.test(lastUserText || '');
  if (!askedPersonalNow) return 0;

  const totalPersonalAsked = countPersonalQuestions(history);
  const { cap, bonus } = personalQuestionsCapByTrust(trustNow);

  if (trustNow < 40) {
    // ранний оффтоп — штраф
    return -4;
  }

  // Если уже исчерпал квоту для текущего trust-бракетa — бонус не даём
  if (totalPersonalAsked > cap) return 0;

  // В пределах лимита — даём бонус за человечность (по правилу бракетов)
  return bonus;
}

// ==== Вежливость с кулдауном (+0.5, не чаще чем раз в 6 ходов) ====

const COURTESY_RX = /(сэр|sir|пожалуйста|плиз|пжл|извините|простите|подскажите|будьте добры)/i;

function courtesyBonusWithCooldown(history=[], lastUserText='') {
  const isPoliteNow = COURTESY_RX.test(String(lastUserText).toLowerCase());
  if (!isPoliteNow) return 0;

  // ищем, когда последний раз встречалась вежливая фраза у пользователя
  const userMsgs = (history || []).filter(m => String(m.role||'').toLowerCase()==='user');
  let lastPoliteIdx = -Infinity;
  for (let i = userMsgs.length - 2; i >= 0; i--) { // -2: исключаем текущую
    const t = (userMsgs[i].text || userMsgs[i].content || userMsgs[i].message || '').toString().toLowerCase();
    if (COURTESY_RX.test(t)) { lastPoliteIdx = i; break; }
  }
  // индекс текущего — userMsgs.length - 1
  const idxNow = userMsgs.length - 1;
  const deltaTurns = idxNow - lastPoliteIdx;

  if (deltaTurns <= COOLDOWN_TURNS) {
    // слишком часто — не даём бонус
    return 0;
  }
  return 0.5; // единичный бонус
}

// ==== Основной расчёт ====

export function computeTrust({ baseTrust = 20, evidences = [], history = [], lastUserText = '' }) {
  let score = clamp01_100(baseTrust);

  // 1) Документы
  const { uniq, hard, med, sup, uniqSet } = countKinds(evidences);
  const hasCoop = uniqSet.has('coop_contract_pdf') || uniqSet.has('contract_pdf');

  score += hard * 10;
  score += Math.min(3, med) * 4;
  score += Math.min(3, sup) * 2;
  if (hasCoop) score += 4;

  if (uniq >= 3) score += 3;
  if (uniq >= 5) score += 3;

  // 2) Стадии
  const stages = (history || []).map(h => String(h.stage || '').toLowerCase());
  const sawContractStage  = stages.includes('contract');
  const sawCandidateStage = stages.includes('candidate');
  const sawPaymentStage   = stages.includes('payment');

  if (sawCandidateStage && (hard + med) > 0) score += 2;
  if (sawContractStage  && hard > 0)         score += 3;

  // 3) Тон/конкретика/фокус
  const tone = toneFromHistory(history);
  score += Math.min(4, tone.polite);
  score += tone.press;
  if (tone.obseq >= 2) score -= 3;

  const concrete = concretenessScore(lastUserText || '');
  const bizFocus = businessFocusScore(lastUserText || '');
  score += concrete;
  score += bizFocus;

  // 4) Красные/зелёные флаги
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

  // 5) Персональные вопросы — по твоим новым правилам
  const personalAdj = personalLifeAdjustment({ trustNow: score, history, lastUserText });
  score += personalAdj;

  // 6) Вежливость с кулдауном
  score += courtesyBonusWithCooldown(history, lastUserText);

  // 7) Микро-кредиты за нормальный диалог (без раннего давления)
  const micro = dialogMicroCredits(history, hard, med);
  score += Math.min(8, micro.uniqInfoKeysCount);
  if (!micro.recentEarlyPayPressure) score += 1;

  // 8) Нелинейные ворота
  if (score > 30 && hard < 1) score = 30; // Gate 1
  const passGate2 = (hard >= 2) || (hard >= 1 && med >= 1);
  if (score > 60 && !passGate2) score = 60; // Gate 2
  if (score > 75 && (hard < 2 || sig.red.length > 0)) score = 75; // Gate 3
  if (sawPaymentStage && score < 75) score = Math.max(0, score - 8);

  return Math.round(clamp01_100(score));

  // ===== Helpers =====
  function clamp01_100(x){ return Math.max(0, Math.min(100, Number(x) || 0)); }

  function dialogMicroCredits(hist=[], hardCount=0, medCount=0) {
    const userMsgs = (hist || []).filter(m => String(m.role||'').toLowerCase()==='user').slice(-12);

    const KEY_REGEXPS = {
      vacancy: /(ваканси|позици|role|position|должност)/i,
      salary: /(зарплат|salary|net|gross|нетто|брутто)/i,
      housing: /(жиль|accommodat|общежит|проживан)/i,
      schedule: /(график|смен|hours|schedule|work\s*time)/i,
      location: /(локац|город|location|where|city)/i,
      documents: /(контракт|офер|demand|документ|agreement)/i,
      website: /(сайт|website|https?:\/\/)/i
    };

    const foundKeys = new Set();
    for (const m of userMsgs) {
      const t = (m.text || m.content || m.message || '').toString();
      for (const [key,rx] of Object.entries(KEY_REGEXPS)) {
        if (rx.test(t)) foundKeys.add(key);
      }
    }

    const payRx = /(оплат|плат[её]ж|инвойс|сч[её]т|реквизит(ы)?)/i;
    const earlyPayHits = userMsgs.slice(-6).filter(m => payRx.test((m.text||m.content||'')+'')).length;
    const pass2 = (hardCount >= 2) || (hardCount >= 1 && medCount >= 1);
    const recentEarlyPayPressure = (earlyPayHits >= 2) && !pass2;

    return {
      uniqInfoKeysCount: foundKeys.size,
      recentEarlyPayPressure
    };
  }
}
