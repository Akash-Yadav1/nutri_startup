from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import pandas as pd
from datetime import datetime
from catboost import CatBoostClassifier

app = Flask(__name__)
CORS(app)

# 🔁 Load CatBoost model
model_path = os.path.join("model", "model.cbm")
model = CatBoostClassifier()
model.load_model(model_path)
model_loaded = True

# 📘 Load nutrition guidance and alternatives
with open("nutri_startup\\Nutrimama-main\\backend\\recommendation\\nutrition_guidelines.json") as f:
    nutrition_guide = json.load(f)

with open("nutri_startup\\Nutrimama-main\\backend\\recommendation\\nutrient_alternatives.json") as f:
    nutrient_map = json.load(f)

os.makedirs("memory", exist_ok=True)

# 🧠 Smart Input Cleaner
def clean_input(data):
    def normalize_number(val):
        if isinstance(val, (int, float)): return val
        val = str(val).lower().strip()
        for word, num in {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }.items():
            if word in val: return num
        try: return float(val)
        except: return 0

    name = data.get("name", "").strip().title()
    age = normalize_number(data.get("age"))
    meals = normalize_number(data.get("meals"))
    water = normalize_number(data.get("water"))

    pregnant = data.get("pregnant", "").lower()
    if "yes" in pregnant or "हां" in pregnant: pregnant = "yes"
    else: pregnant = "no"

    iron = data.get("iron", "").lower()
    if "yes" in iron or "हां" in iron: iron = "yes"
    else: iron = "no"

    income_raw = data.get("income", "").lower()
    if "low" in income_raw: income = "low"
    elif "mid" in income_raw or "medium" in income_raw: income = "mid"
    elif "high" in income_raw: income = "high"
    else: income = "low"

    available_foods = data.get("available_foods", "").lower()

    return {
        "name": name,
        "age": int(age),
        "meals": int(meals),
        "water": float(water),
        "pregnant": pregnant,
        "iron": iron,
        "income": income,
        "available_foods": available_foods
    }

@app.route("/predict", methods=["POST"])
def predict_final():
    try:
        data = request.json
        print("📥 /predict received:", data)

        # Clean input
        cleaned = clean_input(data)

        # Prepare dataframe (categorical support)
        df = pd.DataFrame([{
            "age": cleaned["age"],
            "pregnant": cleaned["pregnant"],
            "meals": cleaned["meals"],
            "water": cleaned["water"],
            "iron": cleaned["iron"],
            "income": cleaned["income"]
        }])

        # Predict using CatBoost
        prediction = model.predict(df)[0]
        label_map = {0: "High", 1: "Low", 2: "Moderate"}
        risk = label_map.get(int(prediction), "Unknown")

        # Get nutrition guidelines
        take = nutrition_guide.get(risk, {}).get("take", [])
        avoid = nutrition_guide.get(risk, {}).get("avoid", [])

        # Alternative foods
        foods = [x.strip() for x in cleaned["available_foods"].split(",")]
        substitutions = []
        for nutrient, d in nutrient_map.items():
            if not any(f in foods for f in d["default"]):
                substitutions.append(f"For {nutrient.upper()}, consider: " + ", ".join(d["alternatives"]))

        # Final response
        result = f"🔍 Hi {cleaned['name']}, your nutritional risk is **{risk.upper()}**.\n\n"
        if take: result += "✅ Foods to TAKE:\n- " + "\n- ".join(take) + "\n\n"
        if avoid: result += "🚫 Foods to AVOID:\n- " + "\n- ".join(avoid) + "\n\n"
        if substitutions: result += "💡 Alternatives:\n- " + "\n- ".join(substitutions)

        # Memory save
        memory_file = os.path.join("memory", f"{cleaned['name'].lower()}.json")
        session = {
            "timestamp": datetime.now().isoformat(),
            "age": cleaned["age"],
            "pregnant": cleaned["pregnant"],
            "risk": risk,
            "available_foods": foods,
            "recommendations": {
                "take": take,
                "avoid": avoid,
                "alternatives": substitutions
            }
        }

        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                memory = json.load(f)
        else:
            memory = {"name": cleaned["name"], "history": []}

        memory["history"].append(session)

        with open(memory_file, "w") as f:
            json.dump(memory, f, indent=2)

        return jsonify({"response": result})

    except Exception as e:
        print("❌ Error in /predict:", str(e))
        return jsonify({"response": f"❌ Error during prediction: {str(e)}"}), 500

@app.route("/history/<username>", methods=["GET"])
def get_history(username):
    try:
        memory_file = os.path.join("memory", f"{username.lower()}.json")
        if not os.path.exists(memory_file):
            return jsonify({"history": []})
        
        with open(memory_file, "r") as f:
            memory = json.load(f)
        
        return jsonify({"history": memory.get("history", [])})
    except Exception as e:
        return jsonify({"error": f"❌ Error fetching history: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
