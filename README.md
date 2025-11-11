# Financial Intelligence System with RAG

A comprehensive financial analysis platform featuring Retrieval-Augmented Generation (RAG), real-time market data integration, and intelligent investment advisory capabilities.

## Overview

The Financial Intelligence System is an advanced AI-powered platform that provides:
- **RAG-Enhanced Q&A**: Retrieval-Augmented Generation with verified financial sources
- **Real-Time Market Analysis**: Live stock data, ratios, and analyst recommendations
- **Portfolio Management**: Diversification analysis and optimization recommendations
- **Market Intelligence**: News aggregation and sentiment analysis
- **Educational Guidance**: Investment strategies with concrete examples

## Key Features

### RAG System Architecture
- **Vector Search**: Semantic document retrieval using FAISS/ChromaDB
- **Knowledge Base**: Automated scraping from SEC, Yahoo Finance, and financial news
- **Source Attribution**: Transparent citation of information sources
- **Fallback Mechanisms**: Graceful degradation when dependencies unavailable

### Data Sources
- **Yahoo Finance**: Real-time prices, company metrics, analyst ratings
- **SEC EDGAR**: Official company filings and reports
- **Financial News**: Bloomberg, Reuters, MarketWatch, CNBC aggregation
- **Analyst Reports**: Investment recommendations and price targets

## Quick Start

### Prerequisites
- Python 3.10 or higher
- OpenAI API key
- 4GB RAM minimum (8GB recommended for full RAG features)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/agentic-web-trust-layer.git
cd agentic-web-trust-layer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
# Copy the example configuration
cp .env.example .env

# Edit .env and add your API keys
OPENAI_API_KEY=your-openai-api-key
FINNHUB_API_KEY=your-finnhub-api-key  # Optional
```

### Running the Application

**Option 1: Automated Start (Recommended)**
```bash
# Unix/macOS
./start.sh

# Or manually start both services
```

**Option 2: Manual Start**

Terminal 1 - Backend API:
```bash
python backend.py
# API runs on http://localhost:8000
```

Terminal 2 - Frontend UI:
```bash
streamlit run app.py
# UI opens at http://localhost:8501
```

## Usage Guide

### 1. Stock Analysis
- Enter any stock symbol (e.g., AAPL, MSFT, GOOGL)
- View comprehensive metrics including P/E ratio, market cap, analyst recommendations
- AI-powered analysis with risk assessment and investment insights

### 2. Portfolio Management
- Add multiple stock holdings with share quantities
- Analyze portfolio diversification using HHI scoring
- Receive AI recommendations for optimization
- Track sector allocation and risk distribution

### 3. Financial Q&A with RAG
- Enable RAG system for enhanced responses
- Ask any financial question (investments, strategies, market analysis)
- View source citations and relevance scores
- Access real-time market data in responses

### 4. Market Intelligence
- Search latest financial news from multiple sources
- Track market sentiment and trends
- Access analyst reports and recommendations

## Example Queries

### Stock Analysis
- "Analyze Apple stock"
- "Compare MSFT vs GOOGL"
- "Tech sector performance"

### Investment Strategy
- "Best dividend stocks for 2024"
- "How to build a retirement portfolio"
- "Growth vs value investing strategies"
- "Risk management techniques"

### Educational Questions
- "Explain P/E ratio with examples"
- "What is dollar-cost averaging?"
- "How do ETFs work?"
- "Understanding market volatility"

## Project Structure

```
agentic-web-trust-layer/
├── app.py              # Streamlit frontend application
├── backend.py          # FastAPI backend server
├── rag_system.py       # RAG implementation with vector search
├── financial_tools.py  # Market data and analysis tools
├── llm_handler.py      # LLM integration and management
├── models.py           # Data models and schemas
├── requirements.txt    # Python dependencies
├── .env.example        # Environment configuration template
├── start.sh            # Startup script for Unix/macOS
├── CHANGELOG.md        # Version history and updates
└── README.md           # Documentation
```

## Configuration

### API Configuration
Edit `.env` file:
```bash
# Required
OPENAI_API_KEY=your-key-here

# Optional for enhanced features
FINNHUB_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
```

### RAG System Settings
Edit `rag_system.py`:
```python
# Choose embedding model
embedding_model = "all-MiniLM-L6-v2"

# Select vector database
use_chromadb = False  # Set to True for ChromaDB
```

### Model Selection
Edit `llm_handler.py`:
```python
# Available models
models = {
    "fast": "gpt-3.5-turbo",
    "quality": "gpt-4",
    "balanced": "gpt-4o-mini"
}
```

## API Documentation

### Core Endpoints

**Stock Analysis**
```bash
GET /stock/{symbol}
GET /stock/{symbol}/history?period=1mo
GET /stock/{symbol}/ratios
```

**Portfolio Management**
```bash
POST /analyze/portfolio
# Body: {"holdings": [{"symbol": "AAPL", "shares": 10}]}
```

**Financial Q&A with RAG**
```bash
POST /query/financial
# Body: {
#   "query": "Your question",
#   "use_rag": true,
#   "risk_tolerance": "moderate"
# }
```

**Market Intelligence**
```bash
GET /market/news?query=tech+stocks&max_results=10
```

## Technical Features

### RAG System
- Vector embeddings with sentence-transformers
- Semantic search using FAISS/ChromaDB
- Document ranking with relevance scoring
- Automatic knowledge base updates
- Fallback to keyword matching when vector search unavailable

### Data Integration
- Real-time market data from Yahoo Finance
- SEC filing scraping and analysis
- Multi-source news aggregation
- Sentiment analysis from headlines
- Portfolio diversification metrics (HHI)

### Performance
- Response caching for improved speed
- Parallel data fetching
- Graceful error handling
- Automatic retry mechanisms

## System Requirements

### Minimum Requirements
- Python 3.10+
- 4GB RAM
- 2GB disk space
- Internet connection

### Recommended Setup
- Python 3.11+
- 8GB RAM
- SSD storage
- GPU for faster embeddings (optional)

## Troubleshooting

### Common Issues

**Backend Connection Error**
- Verify backend is running on port 8000
- Check firewall settings
- Ensure all dependencies installed: `pip install -r requirements.txt`

**OpenAI API Errors**
- Verify API key in `.env` file
- Check API usage limits and billing
- Ensure network connectivity

**RAG System Not Working**
- System gracefully falls back to standard mode
- Check logs for specific dependency issues
- Reinstall vector search dependencies if needed

**Slow Performance**
- Initial RAG indexing takes 1-2 minutes
- Consider reducing document count in `rag_system.py`
- Enable caching in configuration

## Performance Optimization

- **Caching**: Responses cached for 5 minutes
- **Parallel Processing**: Multiple API calls executed simultaneously
- **Lazy Loading**: RAG system initializes on first use
- **Fallback Mechanisms**: Automatic degradation for missing dependencies

## Security Considerations

- API keys stored in environment variables
- No hardcoded credentials in source code  
- Input sanitization for all user queries
- Rate limiting on API endpoints

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with FastAPI, Streamlit, OpenAI GPT-4, and open-source financial data providers.


