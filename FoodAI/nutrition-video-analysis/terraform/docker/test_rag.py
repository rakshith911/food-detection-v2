import sys
sys.path.insert(0, '.')

base = '/Users/rakshithmahishi/Documents/food-detection'
from nutrition_rag_system import NutritionRAG

rag = NutritionRAG(
    fao_faiss_path=f'{base}/fao_data/fao_faiss.index',
    fao_density_path=f'{base}/fao_data/fao_density.json',
    fao_names_path=f'{base}/fao_data/fao_food_names.json',
    usda_faiss_path=f'{base}/usda_data/usda_faiss.index',
    usda_foods_path=f'{base}/usda_data/usda_foods.json',
    usda_names_path=f'{base}/usda_data/usda_food_names.json',
)
rag.load()

print("\n--- RAG lookup results ---")
for food in ['chicken breast', 'pasta', 'rice', 'broccoli', 'salmon', 'cherry tomatoes', 'brussels sprouts', 'celery', 'apple pie', 'butter chicken']:
    d = rag.get_density(food)
    k = rag.get_calories_per_100g(food)
    mass = 200 * d
    kcal = (mass / 100) * k
    print(f'{food:22s}  density={d:.2f}g/ml  kcal/100g={k:.0f}  200ml -> {mass:.0f}g -> {kcal:.0f}kcal')

print("\n--- Full nutrition for 300ml of pasta ---")
result = rag.get_nutrition('pasta', 300.0)
for k, v in result.items():
    print(f'  {k}: {v}')
