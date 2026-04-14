export const toSentenceCase = (value: string | null | undefined): string => {
  const normalized = (value || '').replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return '';
  }

  const lowerCased = normalized.toLowerCase();
  return lowerCased.charAt(0).toUpperCase() + lowerCased.slice(1);
};
