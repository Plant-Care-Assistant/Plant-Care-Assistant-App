// Grouped by disease keyword so one entry covers e.g. "Apple___Black rot" and "Grape___Black rot".

interface AdvicePattern {
  // ALL keywords (lowercase) must appear in the lowercased condition string.
  match: string[];
  advice: string[];
}

const PATTERNS: AdvicePattern[] = [
  {
    match: ['huanglongbing'],
    advice: [
      'No cure — isolate the plant immediately to avoid spreading to other citrus.',
      'Control psyllid insects (the vector) — remove them or use insecticidal soap.',
      'Consider removing severely affected plants to protect your collection.',
    ],
  },
  {
    match: ['mosaic', 'virus'],
    advice: [
      'No cure — viral infection. Isolate the plant from healthy ones right away.',
      'Disinfect tools (pruners, pots) before touching other plants.',
      'If symptoms are severe, remove and discard the plant — do not compost it.',
    ],
  },
  {
    match: ['yellow', 'leaf', 'curl'],
    advice: [
      'No cure — spread by whiteflies. Isolate the plant and control whiteflies.',
      'Remove and discard heavily affected leaves; bag them, do not compost.',
      'Cover nearby healthy plants with insect mesh or move them away.',
    ],
  },
  {
    match: ['late blight'],
    advice: [
      'Aggressive fungal disease — act fast. Remove affected leaves and discard them.',
      'Improve airflow and avoid wetting leaves; water at the soil instead.',
      'Apply a copper-based fungicide on a dry day if the spread is recent.',
    ],
  },
  {
    match: ['early blight'],
    advice: [
      'Prune the lowest, most affected leaves to slow the spread upward.',
      'Mulch the soil to stop spores splashing onto leaves when watering.',
      'Use a copper or chlorothalonil fungicide if the spread continues.',
    ],
  },
  {
    match: ['leaf scorch'],
    advice: [
      'Remove burnt or scorched leaves cleanly with sterilised scissors.',
      'Reduce direct sun if leaves dry out faster than they hydrate.',
      'Check watering — under-watered roots are a common driver of scorch.',
    ],
  },
  {
    match: ['leaf mold'],
    advice: [
      'Increase ventilation and lower humidity around the plant.',
      'Avoid overhead watering — wet leaves let mold spores germinate.',
      'Remove affected leaves and apply a fungicide if it keeps spreading.',
    ],
  },
  {
    match: ['powdery mildew'],
    advice: [
      'Move the plant to a brighter, better-ventilated spot.',
      'Wipe leaves with a 1:9 milk-to-water spray or diluted neem oil weekly.',
      'Remove the worst-affected leaves to slow the spread.',
    ],
  },
  {
    match: ['rust'],
    advice: [
      'Remove all leaves with rust spots — bag them, do not compost.',
      'Water at the soil to keep leaves dry; rust thrives on wet foliage.',
      'Apply a sulphur or copper fungicide if it keeps spreading.',
    ],
  },
  {
    match: ['septoria'],
    advice: [
      'Prune affected leaves promptly; the spots spread upward.',
      'Mulch the soil and water at the base, not on the leaves.',
      'Use a copper-based fungicide if many leaves are involved.',
    ],
  },
  {
    match: ['target spot'],
    advice: [
      'Remove leaves with concentric brown spots — they shed spores.',
      'Space plants and prune for airflow; the fungus likes still humid air.',
      'Use a chlorothalonil or copper fungicide on remaining foliage.',
    ],
  },
  {
    match: ['spider', 'mite'],
    advice: [
      'Rinse leaves (top and underside) with strong water spray — mites hate water.',
      'Increase humidity around the plant; mites prefer dry air.',
      'Apply insecticidal soap or neem oil if rinsing is not enough.',
    ],
  },
  {
    match: ['bacterial spot'],
    advice: [
      'Remove and discard affected leaves on a dry day with clean tools.',
      'Avoid overhead watering — bacteria spread through water splash.',
      'Apply a copper-based bactericide; spacing for airflow also helps.',
    ],
  },
  {
    match: ['black rot'],
    advice: [
      'Prune out affected leaves, stems, or fruit with sterilised tools.',
      'Pick up fallen debris around the plant — it harbours spores.',
      'Apply a fungicide labelled for black rot during humid spells.',
    ],
  },
  {
    match: ['esca'],
    advice: [
      'No cure once advanced — manage by pruning out severely affected wood.',
      'Avoid pruning in wet weather; seal large cuts to keep spores out.',
      'In severe cases, remove the plant to protect other vines.',
    ],
  },
  {
    match: ['scab'],
    advice: [
      'Rake and discard fallen leaves — the fungus overwinters on them.',
      'Improve airflow with careful pruning to dry leaves faster.',
      'Apply a fungicide at bud-break and again 1–2 weeks later.',
    ],
  },
  {
    match: ['blight'],
    advice: [
      'Remove visibly infected leaves to slow the spread.',
      'Water at the soil — fungal blights move through wet foliage.',
      'A copper-based fungicide helps if applied early.',
    ],
  },
  {
    match: ['cercospora'],
    advice: [
      'Strip the most-spotted leaves to reduce inoculum.',
      'Improve airflow and avoid overhead watering.',
      'Apply a fungicide labelled for leaf spot if it keeps progressing.',
    ],
  },
];

export function getDiseaseAdvice(condition: string): string[] | null {
  if (!condition) return null;
  const c = condition.toLowerCase();
  for (const pattern of PATTERNS) {
    if (pattern.match.every((kw) => c.includes(kw))) {
      return pattern.advice;
    }
  }
  return null;
}
