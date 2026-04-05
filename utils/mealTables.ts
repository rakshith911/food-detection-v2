import type { NutritionItem } from '../services/NutritionAnalysisAPI';
import type {
  DishContent,
  DishTableKey,
  DishTableSection,
  QuestionnaireContext,
  QuestionnaireIngredient,
} from '../store/slices/historySlice';

const TABLE_TITLES: Record<DishTableKey, string> = {
  base: 'Base Kcal Table',
  highCalorie: 'High Calorie Content Table',
  hiddenContent: 'Hidden Content Table',
};

const TABLE_ORDER: DishTableKey[] = ['base', 'highCalorie', 'hiddenContent'];

const normalizeText = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

const isIngredientMatch = (foodName: string, ingredientName: string) => {
  const normalizedFood = normalizeText(foodName);
  const normalizedIngredient = normalizeText(ingredientName);

  if (!normalizedFood || !normalizedIngredient) {
    return false;
  }

  return (
    normalizedFood === normalizedIngredient ||
    normalizedFood.includes(normalizedIngredient) ||
    normalizedIngredient.includes(normalizedFood)
  );
};

const matchesQuestionnaireList = (
  foodName: string,
  items: QuestionnaireIngredient[] | undefined
) => (items || []).some((item) => isIngredientMatch(foodName, item.name || ''));

const createRow = (
  name: string,
  mass_g: number | undefined,
  total_calories: number | undefined,
  index: number,
  prefix: DishTableKey
): DishContent => ({
  id: `${prefix}_${Date.now()}_${index}`,
  name: name || 'Unknown Food',
  weight: mass_g && Math.round(mass_g) > 0 ? Math.round(mass_g).toString() : '',
  calories: Math.round(total_calories || 0).toString(),
});

const createRowFromItem = (item: NutritionItem, index: number, prefix: DishTableKey): DishContent =>
  createRow(item.food_name || 'Unknown Food', item.mass_g, item.total_calories, index, prefix);

export const createEmptyDishTables = (): DishTableSection[] =>
  TABLE_ORDER.map((key) => ({
    key,
    title: TABLE_TITLES[key],
    rows: [],
  }));

export const hydrateDishTables = (
  dishTables?: DishTableSection[],
  dishContents?: DishContent[]
): DishTableSection[] => {
  const byKey = new Map<DishTableKey, DishTableSection>();

  (dishTables || []).forEach((section) => {
    byKey.set(section.key, {
      key: section.key,
      title: section.title || TABLE_TITLES[section.key],
      rows: section.rows || [],
    });
  });

  if (!byKey.has('base') && dishContents?.length) {
    byKey.set('base', {
      key: 'base',
      title: TABLE_TITLES.base,
      rows: dishContents,
    });
  }

  return TABLE_ORDER.map((key) => ({
    key,
    title: byKey.get(key)?.title || TABLE_TITLES[key],
    rows: byKey.get(key)?.rows || [],
  }));
};

export const buildDishTablesFromItems = (
  items: NutritionItem[] = [],
  questionnaireContext?: QuestionnaireContext
): DishTableSection[] => {
  const tables = createEmptyDishTables();
  let derivedIndex = 0;

  const pushRow = (
    targetKey: DishTableKey,
    row: DishContent
  ) => {
    const section = tables.find((table) => table.key === targetKey);
    section?.rows.push(row);
  };

  items.forEach((item, index) => {
    const foodName = item.food_name || '';
    const hasComponentBreakdown =
      !!item.base_component ||
      !!item.extra_component ||
      !!item.hidden_component;

    if (hasComponentBreakdown) {
      if (item.base_component) {
        pushRow(
          'base',
          createRow(
            foodName,
            item.base_component.mass_g,
            item.base_component.total_calories,
            derivedIndex++,
            'base'
          )
        );
      }

      if (item.extra_component) {
        pushRow(
          'highCalorie',
          createRow(
            foodName,
            item.extra_component.mass_g,
            item.extra_component.total_calories,
            derivedIndex++,
            'highCalorie'
          )
        );
      }

      if (item.hidden_component) {
        pushRow(
          'hiddenContent',
          createRow(
            foodName,
            item.hidden_component.mass_g,
            item.hidden_component.total_calories,
            derivedIndex++,
            'hiddenContent'
          )
        );
      }
      return;
    }

    // Trust the backend's role_tag when available — avoids incorrectly moving base items
    // (e.g. "Chicken breast") into hidden just because they share a name with a questionnaire entry.
    const targetKey: DishTableKey =
      item.role_tag === 'hidden' ? 'hiddenContent' :
      item.role_tag === 'high_calorie' ? 'highCalorie' :
      item.role_tag === 'base' ? 'base' :
      matchesQuestionnaireList(foodName, questionnaireContext?.hidden_ingredients) ? 'hiddenContent' :
      matchesQuestionnaireList(foodName, questionnaireContext?.extras) ? 'highCalorie' :
      'base';

    pushRow(targetKey, createRowFromItem(item, index, targetKey));
  });

  return tables;
};

export const getTableTotals = (rows: DishContent[]) => {
  const totalWeight = rows.reduce((sum, row) => sum + (Number(row.weight) || 0), 0);
  const totalCalories = rows.reduce((sum, row) => sum + (Number(row.calories) || 0), 0);

  return {
    totalWeight,
    totalCalories,
  };
};

export const getOverallCaloriesFromTables = (dishTables: DishTableSection[]) =>
  dishTables.reduce((sum, section) => sum + getTableTotals(section.rows).totalCalories, 0);

export const getBaseDishContents = (dishTables: DishTableSection[]) =>
  dishTables.find((section) => section.key === 'base')?.rows || [];

export const getMealNameFromTables = (
  dishTables: DishTableSection[],
  fallback: string = 'Analyzed Meal'
) => {
  const firstBase = getBaseDishContents(dishTables).find((row) => row.name.trim());
  if (firstBase) return firstBase.name;

  const firstAny = dishTables.flatMap((section) => section.rows).find((row) => row.name.trim());
  return firstAny?.name || fallback;
};
