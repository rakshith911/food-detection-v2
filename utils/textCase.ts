export const toSentenceCase = (value: string | null | undefined): string => {
  const normalized = (value || '').replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return '';
  }

  const lowerCased = normalized.toLowerCase();
  return lowerCased.charAt(0).toUpperCase() + lowerCased.slice(1);
};

const normalizeDisplayLabelKey = (value: string | null | undefined): string =>
  (value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const LEADING_PREP_WORDS = new Set([
  'chopped',
  'diced',
  'minced',
  'sliced',
  'shredded',
  'grated',
  'julienned',
  'cubed',
  'finely',
  'roughly',
  'thinly',
  'thickly',
]);

export const toDisplayFoodLabel = (value: string | null | undefined): string => {
  const normalized = normalizeDisplayLabelKey(value);

  if (!normalized) {
    return '';
  }

  const tokens = normalized.split(' ').filter(Boolean);
  let firstIngredientIndex = 0;

  while (
    firstIngredientIndex < tokens.length - 1 &&
    LEADING_PREP_WORDS.has(tokens[firstIngredientIndex])
  ) {
    firstIngredientIndex += 1;
  }

  const cleaned = tokens.slice(firstIngredientIndex).join(' ') || normalized;
  return toSentenceCase(cleaned);
};
