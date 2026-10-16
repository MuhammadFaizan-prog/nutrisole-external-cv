// Maps raw detector class names to (a) a readable label and (b) the food items that already exist in NutriSole's
// catalogue (src/nutrisole/extensions/model.ts: Apple, Banana, Tomato). Anything else is shown as "recognised, but not
// in NutriSole's food library yet" with no nutrition or price.

export const APP_FOODS = ['Apple', 'Banana', 'Tomato'] as const;
export type AppFood = (typeof APP_FOODS)[number];

const TO_APP_FOOD: Record<string, AppFood> = {
  apple: 'Apple',
  banana: 'Banana',
  tomato: 'Tomato',
};

/** First alias of a LVIS-style name ("bell pepper/capsicum" -> "bell pepper"), lower-cased and trimmed. */
export function canonicalName(raw: string): string {
  return raw.split('/')[0].trim().toLowerCase();
}

/** Human label: "bell pepper/capsicum" -> "Bell pepper", "orange/orange fruit" -> "Orange". */
export function displayLabel(raw: string): string {
  const name = canonicalName(raw);
  return name ? name[0].toUpperCase() + name.slice(1) : 'Unknown';
}

/** The NutriSole food a detector class corresponds to, or null when the app has no catalogue entry for it. */
export function appFoodFor(raw: string): AppFood | null {
  return TO_APP_FOOD[canonicalName(raw)] ?? null;
}

/** Class indices of a model's label list that map to each app food (several when a model repeats a class). */
export function appFoodIndex(names: readonly string[]): Record<AppFood, number[]> {
  const out = { Apple: [], Banana: [], Tomato: [] } as Record<AppFood, number[]>;
  names.forEach((raw, i) => { const food = appFoodFor(raw); if (food) out[food].push(i); });
  return out;
}
