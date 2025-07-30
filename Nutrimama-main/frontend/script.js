const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");

let step = 0;
const userData = {};
const questions = [
    "What is your name?",
    "What is your age?",
    "Are you pregnant? (yes/no)",
    "How many meals do you eat per day?",
    "How much water (in liters) do you drink daily?",
    "Do you take iron tablets? (yes/no)",
    "What is your income level? (low/mid/high)",
    "Which foods are available to you regularly? (e.g., rice, dal, milk, eggs, ragi, fruits)"
];

function addMessage(text, sender) {
    const div = document.createElement("div");
    div.className = sender;
    div.textContent = `${sender === "user" ? "You: " : "Bot: "}${text}`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function nextQuestion() {
    if (step < questions.length) {
        addMessage(questions[step], "bot");
    } else {
        sendForPrediction();
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, "user");

    switch (step) {
        case 0: userData.name = message; break;
        case 1: userData.age = parseInt(message); break;
        case 2: userData.pregnant = message.toLowerCase(); break;
        case 3: userData.meals = parseInt(message); break;
        case 4: userData.water = parseFloat(message); break;
        case 5: userData.iron = message.toLowerCase(); break;
        case 6: userData.income = message.toLowerCase(); break;
        case 7: userData.available_foods = message.toLowerCase(); break;
    }

    userInput.value = "";
    step++;
    nextQuestion();
}

async function sendForPrediction() {
    console.log("Sending to /predict:", userData);
    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(userData)
        });

        if (!response.ok) throw new Error("Prediction failed");

        const data = await response.json();
        addMessage(data.response, "bot");

        // Reset state
        step = 0;
        for (const key in userData) delete userData[key];
        addMessage("Let's start a new session. What is your name?", "bot");
    } catch (error) {
        addMessage("❌ Could not connect to NutriMama backend. Is it running?", "bot");
    }
}

async function viewHistory() {
    if (!userData.name) {
        addMessage("❗ Please provide your name first.", "bot");
        return;
    }

    try {
        const res = await fetch(`http://127.0.0.1:5000/history/${userData.name.toLowerCase()}`);
        if (!res.ok) throw new Error("No history found");

        const data = await res.json();
        if (!data.history || data.history.length === 0) {
            addMessage("No history found yet.", "bot");
            return;
        }

        addMessage("📚 Here's your past nutrition history:", "bot");
        data.history.forEach(entry => {
            addMessage(`🕓 ${new Date(entry.timestamp).toLocaleString()}\nRisk: ${entry.risk}\nFoods: ${entry.available_foods.join(", ")}`, "bot");
        });

    } catch (err) {
        addMessage("⚠️ Could not load history.", "bot");
    }
}

window.onload = () => {
    addMessage("Hi! I'm NutriMama. Let's assess your nutrition.", "bot");
    step = 0;
    nextQuestion();
};

userInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") sendMessage();
});
