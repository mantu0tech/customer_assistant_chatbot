

# 🤖 AI Customer Support Chatbot

An **AI-powered Customer Support Chatbot** built using **Sentence Transformers, FAISS, and Groq LLMs**.
This project uses **semantic search** to find the most relevant customer support response and then **rephrases it using an LLM** to make it more human, friendly, and professional.

---

## 📌 Features

* 🔍 **Semantic Search** using Sentence Transformers
* ⚡ **Fast similarity matching** with FAISS
* 🧠 **LLM-based response rephrasing** using Groq
* 📚 Trained on **Bitext Customer Support Dataset (27K responses)**
* 🖥️ CLI-based chatbot (easy to extend to Streamlit / Web)
* 🔐 Secure API key handling using environment variables

---

## 🧠 How It Works

1. Customer enters a query
2. Query is converted into an embedding
3. FAISS finds the closest matching support instruction
4. The matched response is sent to Groq LLM
5. LLM rewrites the response in a natural, conversational way

---

## 🖼️ AI Illustration

![AI Support Bot](https://images.unsplash.com/photo-1677442136019-21780ecad995)

---

## 📁 Project Structure

```
customer_support_chatbot/
│
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
```

---

## ⚙️ Requirements

* Python **3.10**
* Windows / Linux / macOS
* Internet connection (for model download & Groq API)

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/customer_support_chatbot.git
cd customer_support_chatbot
```

---

### 2️⃣ Create Virtual Environment (Recommended)

**Windows:**

```powershell
python -m venv new
new\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Groq API Key

⚠️ **Do NOT hardcode API keys in code**

**Windows (PowerShell):**

```powershell
setx GROQ_API_KEY "your_groq_api_key_here"
```

**Linux / macOS:**

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Restart terminal after setting the key.

---

## ▶️ Run the Application

```bash
python app.py
```

You should see:

```
✅ Customer Support Chatbot Ready!
Type 'exit' to quit.
```

---

## 💬 Sample Interaction

```
You: I want to reset my password

Closest Matched Instruction:
→ How can I reset my account password?

Bot Response:
→ Sure! You can reset your password by clicking on the "Forgot Password" link on the login page...
```

---

## 🚀 Future Enhancements

* 🌐 Streamlit Web UI
* 🐳 Docker support
* ☁️ AWS / EC2 deployment
* 💾 Persistent FAISS index
* 🧾 Chat history memory

---

## 🛡️ Security Best Practices

* Keep API keys in **environment variables**
* Do not commit `.env` files
* Rotate keys regularly

---

## 📜 License

This project is for **educational and portfolio purposes**.

---

## 🙌 Author

**Ansari Mantasha**
Cloud & DevOps | AI Enthusiast | Trainer

---

⭐ If you like this project, don’t forget to **star the repository**!
