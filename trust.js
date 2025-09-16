// trust.js
export function computeTrust({ baseTrust = 30, evidences = [], history = [] }) {
  // evidences: массив строк, напр. ["business_card","contract","visa_example","demand_letter"]
  let trust = baseTrust;

  const weights = {
    business_card: 10,
    demand_letter: 20,
    contract: 30,
    visa_example: 25,
    company_website: 10,
    verified_registry: 20
  };

  const unique = new Set(evidences);
  for (const ev of unique) trust += (weights[ev] || 0);

  // Бонус за последовательность стадий (менеджер не прыгает хаотично)
  const stagesOrder = ["Greeting","Demand","Candidate","Contract","Payment","Closing"];
  const lastStages = history.slice(-4).map(h => h.stage);
  const inOrder = lastStages.every((s, i, arr) => i === 0 || stagesOrder.indexOf(arr[i-1]) <= stagesOrder.indexOf(s));
  if (inOrder && lastStages.length >= 3) trust += 10;

  // Ограничители
  trust = Math.max(0, Math.min(100, trust));
  return trust;
}