Nutrimama_langmem/
├── backend/
│   ├── app.py                    ← 🧠 Main Flask API (chat + prediction)
│   ├── memory_store.json         ← 💾 Session memory store (optional, auto-created)
│   ├── requirements.txt          ← 📦 All dependencies (Flask, joblib, scikit-learn, etc.)
│   └── model/                    ← 🤖 Trained model assets
│       ├── model.pkl             ← Your trained ML model (LogisticRegression, etc.)
│       └── vectorizer.pkl        ← Your text vectorizer (Tfidf/CountVectorizer)
│
├── frontend/
│   ├── index.html                ← 🌐 Web UI (chat interface)
│   ├── style.css                 ← 🎨 Styling
│   └── script.js                 ← ⚙️ Frontend chat logic (calls Flask backend)
│
└── README.md                     ← 📖 Project description, setup, usage