"""
nutrition_data.py
-----------------
Nutrition Database and Helper Functions for Healthy Food Detection App.

Provides nutritional details, health categorizations, calorie estimates,
and key benefit notes for supported food classes.
"""

from typing import Dict, Any

# Nutritional Database for food classification
# Keys match expected label names from TensorFlow model (case-insensitive)
FOOD_DATABASE: Dict[str, Dict[str, Any]] = {
    "apple": {
        "name": "Apple",
        "category": "Healthy",
        "calories": "95 kcal (per medium apple)",
        "benefits": "Line 1: High in dietary fiber (pectin), Vitamin C, and antioxidants (quercetin).\nLine 2: Promotes cardiovascular health, aids gut digestion, and assists with blood sugar regulation.",
        "icon": "🍎"
    },
    "banana": {
        "name": "Banana",
        "category": "Healthy",
        "calories": "105 kcal (per medium banana)",
        "benefits": "Line 1: Rich source of essential Potassium, Vitamin B6, and fast-acting complex carbohydrates.\nLine 2: Supports optimal muscle recovery, regulates fluid balance, and sustains physical endurance.",
        "icon": "🍌"
    },
    "orange": {
        "name": "Orange",
        "category": "Healthy",
        "calories": "62 kcal (per medium orange)",
        "benefits": "Line 1: Packed with over 100% daily Vitamin C requirement, Folate, and natural hydration.\nLine 2: Strengthens immune response, enhances skin collagen synthesis, and fights cellular oxidative stress.",
        "icon": "🍊"
    },
    "pizza": {
        "name": "Pizza",
        "category": "Unhealthy",
        "calories": "285 kcal (per slice, approx. 107g)",
        "benefits": "Line 1: High in saturated fat, sodium, and refined carbohydrates with low dietary fiber.\nLine 2: Frequent consumption can increase risks of elevated cholesterol, blood pressure spikes, and weight gain.",
        "icon": "🍕"
    },
    "burger": {
        "name": "Burger",
        "category": "Unhealthy",
        "calories": "354 kcal (per single patty burger)",
        "benefits": "Line 1: Dense in calorie count, trans/saturated fats, and refined wheat flour.\nLine 2: Provides protein but lacks crucial micronutrients; excessive intake is linked to metabolic stress.",
        "icon": "🍔"
    },
    "chips": {
        "name": "Chips",
        "category": "Unhealthy",
        "calories": "152 kcal (per 28g / small bag)",
        "benefits": "Line 1: Ultra-processed snack fried in industrial oils with excessive sodium chloride.\nLine 2: Contains acrylamides and empty calories that disrupt appetite regulation without providing lasting satiety.",
        "icon": "🍟"
    }
}


def get_food_info(food_name: str) -> Dict[str, Any]:
    """
    Safely queries the nutrition database for a given food item name.
    
    Args:
        food_name (str): Raw string predicted by classifier (e.g. "0 Apple" or "banana")
        
    Returns:
        Dict[str, Any]: Dictionary containing name, category, calories, benefits, and icon.
    """
    if not food_name:
        return _get_unknown_food("Unknown")

    # Clean input label (remove index prefixes like "0 Apple" -> "Apple")
    cleaned_name = food_name.strip()
    parts = cleaned_name.split(maxsplit=1)
    if len(parts) > 1 and parts[0].isdigit():
        cleaned_name = parts[1]

    lookup_key = cleaned_name.lower().strip()

    # Search in database
    if lookup_key in FOOD_DATABASE:
        return FOOD_DATABASE[lookup_key]

    # Partial substring search fallback
    for key, data in FOOD_DATABASE.items():
        if key in lookup_key or lookup_key in key:
            return data

    # Default fallback for unrecognized foods
    return _get_unknown_food(cleaned_name.capitalize())


def _get_unknown_food(name: str) -> Dict[str, Any]:
    """Returns fallback dictionary for unlisted food items."""
    return {
        "name": name if name else "Unrecognized Item",
        "category": "Unknown",
        "calories": "N/A",
        "benefits": "Food item detected, but nutritional details are not available in the local database.",
        "icon": "❓"
    }


if __name__ == "__main__":
    # Self-test execution
    test_items = ["Apple", "00 Pizza", "02 Banana", "Chips", "Salad"]
    for item in test_items:
        info = get_food_info(item)
        print(f"[{item}] -> {info['name']} | {info['category']} | {info['calories']}")
