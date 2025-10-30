# 🚀 AgentCert MVP - Quick Start Guide

## ⚡ Super Quick Start (3 steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Your API Key
Create a `.env` file (copy from `.env.example`):
```bash
# If on Windows, run:
copy .env.example .env

# If on Mac/Linux, run:
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Run the System

**Option A: Use the batch files (Windows):**
- Double-click `start_backend.bat` to start the API
- Double-click `start_frontend.bat` to start the UI

**Option B: Use command line:**
Open **Terminal 1**:
```bash
uvicorn backend:app --reload
```

Open **Terminal 2**:
```bash
streamlit run app.py
```

## 🎯 First Test Run

1. The Streamlit app will open in your browser at `http://localhost:8501`
2. You should see "✅ Backend API connected" message
3. Select an example question or enter your own
4. Click "▶️ Run Task"
5. Wait for agent response
6. Click "📊 Evaluate Response"
7. Review the scores
8. Click "✨ Generate Certification"

## 💡 Pro Tips

- The certification requires **85% or higher** in all three categories (accuracy, clarity, compliance)
- Try running multiple iterations to see improvement
- Use the "Clear History" button to start fresh
- Check the "📊 Evaluation History" section for trend visualization

## 🐛 Troubleshooting

**"Backend API is not running"**
- Make sure Terminal 1 is running `uvicorn backend:app --reload`
- The backend should show: `Application startup complete.`

**"OpenAI API key error"**
- Check that your `.env` file exists and has `OPENAI_API_KEY=sk-...`
- Make sure there's no extra spaces in the `.env` file

**Port already in use?**
- Backend is on port 8000, Frontend on 8501
- Close other applications using these ports
- Or change ports in the respective files

## 📚 Next Steps

- Try different financial scenarios
- Experiment with multiple iterations
- Explore the evaluation feedback
- Generate your first certification!

---

**Happy Certifying! 🎓**


