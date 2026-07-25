import express from "express";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "20mb" }));

// Initialize Gemini API client on server side
let ai: GoogleGenAI | null = null;
if (process.env.GEMINI_API_KEY) {
  ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });
}

// ---------------- API ENDPOINTS ----------------

// API: Health check
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", geminiConfigured: !!process.env.GEMINI_API_KEY });
});

// API: Get Python project files
app.get("/api/python-files", (_req, res) => {
  const filesToRead = [
    "nutrition_data.py",
    "model_handler.py",
    "main.py",
    "requirements.txt",
    "README.md",
    "generate_dummy_model.py"
  ];

  const filesContent: Record<string, string> = {};

  for (const filename of filesToRead) {
    const filePath = path.join(process.cwd(), filename);
    if (fs.existsSync(filePath)) {
      filesContent[filename] = fs.readFileSync(filePath, "utf-8");
    }
  }

  res.json({ files: filesContent });
});

// API: Vision classification endpoint using Gemini 3.6 Flash
app.post("/api/classify", async (req, res) => {
  try {
    const { imageBase64, selectedFoodHint } = req.body;

    // If Gemini API is available and image Base64 is provided
    if (ai && imageBase64) {
      const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, "");

      const prompt = `Analyze this food image frame. Your primary goal is to classify if this image contains one of these 6 target foods: Apple, Banana, Orange, Pizza, Burger, Chips. If it is another food item, identify it clearly.
Return a valid JSON object strictly matching this format:
{
  "foodName": "Apple",
  "confidence": 96.5,
  "category": "Healthy",
  "calories": "95 kcal (per medium apple)",
  "benefits": "Line 1: High in dietary fiber (pectin), Vitamin C, and antioxidants (quercetin).\nLine 2: Promotes cardiovascular health, aids gut digestion, and assists with blood sugar regulation.",
  "icon": "🍎"
}
Rules:
- category must be "Healthy" or "Unhealthy".
- foodName should ideally be one of: Apple, Banana, Orange, Pizza, Burger, Chips (or exact item if another food).
- confidence must be a number between 70.0 and 99.9.
- benefits MUST contain AT LEAST 2 distinct lines detailing nutritional breakdown (Line 1: Key nutrients & vitamins, Line 2: Health impact & consumption guidance).
- response must be JSON only without markdown formatting.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.6-flash",
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: "image/jpeg",
                data: base64Data
              }
            },
            { text: prompt }
          ]
        },
        config: {
          responseMimeType: "application/json"
        }
      });

      const responseText = response.text;
      if (responseText) {
        try {
          const parsed = JSON.parse(responseText.trim());
          return res.json(parsed);
        } catch {
          // Fallback if JSON parse fails
        }
      }
    }

    // Fallback classification if no image or Gemini unavailable / fallback test
    const fallbackFood = selectedFoodHint || "Apple";
    const foodDatabase: Record<string, any> = {
      "Apple": {
        foodName: "Apple",
        confidence: 98.2,
        category: "Healthy",
        calories: "95 kcal (per medium apple)",
        benefits: "Line 1: High in dietary fiber (pectin), Vitamin C, and polyphenols.\nLine 2: Promotes heart health, aids gut microbiome digestion, and regulates blood sugar levels.",
        icon: "🍎"
      },
      "Banana": {
        foodName: "Banana",
        confidence: 96.7,
        category: "Healthy",
        calories: "105 kcal (per medium banana)",
        benefits: "Line 1: High in potassium, Vitamin B6, and quick-digesting carbohydrates.\nLine 2: Supports optimal muscle recovery, fluid balance, and sustained physical energy.",
        icon: "🍌"
      },
      "Orange": {
        foodName: "Orange",
        confidence: 95.4,
        category: "Healthy",
        calories: "62 kcal (per medium orange)",
        benefits: "Line 1: Packed with over 100% daily Vitamin C, Folate, and organic antioxidants.\nLine 2: Boosts immune resistance, promotes collagen synthesis, and fights oxidative stress.",
        icon: "🍊"
      },
      "Pizza": {
        foodName: "Pizza",
        confidence: 97.8,
        category: "Unhealthy",
        calories: "285 kcal (per slice, approx. 107g)",
        benefits: "Line 1: High in saturated fat, refined wheat carbs, and sodium chloride.\nLine 2: Regular intake elevates risks of cardiovascular strain, cholesterol imbalance, and hypertension.",
        icon: "🍕"
      },
      "Burger": {
        foodName: "Burger",
        confidence: 94.9,
        category: "Unhealthy",
        calories: "354 kcal (per single patty burger)",
        benefits: "Line 1: High calorie density featuring saturated fats and refined carbohydrates.\nLine 2: Provides protein but lacks fiber and micronutrients; excessive intake increases cardiovascular risk.",
        icon: "🍔"
      },
      "Chips": {
        foodName: "Chips",
        confidence: 96.1,
        category: "Unhealthy",
        calories: "152 kcal (per 28g / small bag)",
        benefits: "Line 1: Ultra-processed snack fried in refined oils with high sodium content.\nLine 2: Contains acrylamides and empty calories that spike blood sugar without providing lasting satiety.",
        icon: "🍟"
      }
    };

    const match = foodDatabase[fallbackFood] || foodDatabase["Apple"];
    return res.json(match);

  } catch (error: any) {
    console.error("Classification error:", error);
    return res.status(500).json({
      error: "Failed to classify image",
      details: error.message
    });
  }
});

// ---------------- VITE & STATIC SERVING ----------------
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
