# 🤖 AgentCert MVP

A trust and certification system for AI agents - specifically demonstrating a **Financial Advisor Agent** that can be evaluated and certified based on accuracy, clarity, and compliance.

## 🎯 Overview

AgentCert MVP allows you to:
1. ✅ Test an AI agent on financial reasoning tasks
2. 🔍 Automatically evaluate responses for accuracy, clarity, and compliance
3. 📊 Track improvement across iterations
4. 🏆 Generate verifiable JSON certifications when benchmarks are met
5. 📈 Visualize performance via a Streamlit dashboard

## 🏗️ Architecture

### Backend (FastAPI)
- **`/run_task`** - Generate agent responses using GPT-4
- **`/evaluate`** - Score responses on accuracy, clarity, compliance
- **`/certify`** - Issue certification JSON when threshold is met (85%)
- **`/history`** - Get evaluation history
- **`DELETE /history`** - Clear history

### Frontend (Streamlit)
- 📝 Task input with example financial scenarios
- 🤖 Agent response display
- 📊 Real-time evaluation scores with progress bars
- 💬 Detailed feedback from evaluator
- 🏆 Certification generation
- 📈 Performance history and trend visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd E:\Cursor\agentcert0
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here
```

### Running the System

You need to run **both the backend and frontend** in separate terminals.

#### Terminal 1 - Backend API:
```bash
uvicorn backend:app --reload
```

The backend will run on `http://localhost:8000`

#### Terminal 2 - Frontend UI:
```bash
streamlit run app.py
```

The UI will automatically open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Start both services** (backend + frontend)

2. **Run a Task:**
   - Select an example question or enter your own financial scenario
   - Click "▶️ Run Task"
   - Wait for the agent to generate a response

3. **Evaluate:**
   - Click "📊 Evaluate Response"
   - Review the scores for accuracy, clarity, and compliance
   - Read the detailed feedback

4. **Generate Certification:**
   - Click "✨ Generate Certification"
   - If the average score is above 85% in all categories, you'll receive a certification badge and JSON

5. **View History:**
   - Scroll down to see all evaluation attempts
   - View trend charts showing improvement over time

## 📝 Example Questions

The app includes these predefined financial scenarios:
- "How should I diversify a $10K portfolio?"
- "What are the risks of investing in a single tech stock?"
- "Explain compound interest to a beginner."
- "What is dollar-cost averaging?"
- "How does an emergency fund help with financial security?"

## 🎯 Evaluation Criteria

Agents are evaluated on three dimensions (0-100 each):

1. **Accuracy** - Factual correctness and financial knowledge
2. **Clarity** - Ease of understanding and structure
3. **Compliance** - Educational focus, avoiding specific investment advice

**Certification Threshold:** 85% or higher in all categories

## 📦 Project Structure

```
agentcert0/
├── app.py              # Streamlit frontend
├── backend.py          # FastAPI backend
├── models.py           # Pydantic data models
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your API keys (not in repo)
└── README.md           # This file
```

## 🔧 Configuration

### Adjusting the Certification Threshold
Edit `backend.py` in the `/certify` endpoint:
```python
threshold = 85.0  # Change this value
```

### Changing the Agent Model
Edit `backend.py` in the `/run_task` endpoint:
```python
model="gpt-4o-mini"  # Change to "gpt-4" or other models
```

### Modifying Evaluation Criteria
Edit the `eval_prompt` in the `/evaluate` endpoint in `backend.py`

## 🧪 Testing

Test the backend API directly:
```bash
# Start backend
uvicorn backend:app --reload

# In another terminal, test endpoints
curl -X POST http://localhost:8000/run_task -H "Content-Type: application/json" -d '{"prompt":"What is compound interest?"}'

curl -X POST http://localhost:8000/evaluate -H "Content-Type: application/json" -d '{"prompt":"...","response":"..."}'

curl -X POST http://localhost:8000/certify
```

## 🎨 Features

- ✅ Real-time agent response generation
- ✅ Automated evaluation using GPT-4
- ✅ Visual score display with progress bars
- ✅ Detailed feedback on each response
- ✅ Certification system with JSON export
- ✅ Performance history tracking
- ✅ Trend visualization
- ✅ Clear history functionality

## 🚧 Future Enhancements

- SQLite database for persistent storage
- Multiple agent types beyond Financial Advisor
- Customizable evaluation rubrics
- Export certification as downloadable JSON
- Advanced analytics dashboard
- API rate limiting and authentication
- Multi-agent comparison tools

## 🤝 Contributing

This is an MVP (Minimum Viable Product). Feel free to extend and customize for your needs!

## 📄 License

MIT License - Feel free to use this for your projects.

## 🐛 Troubleshooting

**Issue:** Backend not connecting
- Ensure the backend is running on port 8000
- Check that no firewall is blocking the connection

**Issue:** OpenAI API errors
- Verify your API key in `.env`
- Check your OpenAI account balance
- Ensure you have API access enabled

**Issue:** Streamlit app not loading
- Check that Streamlit is installed: `pip install streamlit`
- Try accessing the app manually at `http://localhost:8501`

## 💡 Tips

- Start with example questions to see the system in action
- Try multiple iterations to show the feedback loop
- The certification requires consistent high scores across all three categories
- Use the "Clear History" button to start fresh

---

Built with ❤️ using FastAPI, Streamlit, and OpenAI GPT-4


