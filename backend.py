"""
Enhanced FastAPI backend with Groq LLM, financial analysis, and web search
"""
import os
import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime
import logging

# Import custom modules
from models import (
    AgentTask, AgentResponse, Evaluation, Certification, 
    EvaluationHistory
)
from llm_handler import llm_handler, LLMProvider
from financial_tools import financial_analyzer, portfolio_analyzer
from rag_system import get_rag_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Financial Intelligence System",
    description="Advanced financial analysis with Groq LLM, real-time data, and web search",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
evaluation_history = EvaluationHistory()
cache = {}  # Simple cache for expensive operations


# Enhanced Pydantic models
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    analysis_type: str = Field("comprehensive", description="Type of analysis")
    use_groq: bool = Field(True, description="Use Groq for ultra-fast response")


class PortfolioRequest(BaseModel):
    holdings: List[Dict[str, float]] = Field(..., description="List of holdings with symbol and shares")
    analyze: bool = Field(True, description="Perform AI analysis")


class FinancialQueryRequest(BaseModel):
    query: str = Field(..., description="Financial query or question")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    use_web_search: bool = Field(True, description="Include web search results")
    use_rag: bool = Field(True, description="Use RAG system for enhanced responses")
    risk_tolerance: str = Field("moderate", description="Risk tolerance level")
    provider: Optional[str] = Field("openai", description="LLM provider to use")


class MarketNewsRequest(BaseModel):
    query: str = Field(..., description="News search query")
    max_results: int = Field(10, description="Maximum results to return")


# API Endpoints

@app.get("/")
async def root():
    """Enhanced system status"""
    providers = llm_handler.get_available_providers()
    return {
        "message": "Financial Intelligence System",
        "status": "operational",
        "version": "2.0.0",
        "features": {
            "llm_providers": providers,
            "groq_available": "groq" in providers,
            "financial_data": "yfinance",
            "web_search": "duckduckgo",
            "real_time_analysis": True
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "groq": "groq" in llm_handler.get_available_providers(),
            "openai": "openai" in llm_handler.get_available_providers(),
            "financial_data": True,
            "web_search": True
        }
    }
    return health_status


@app.post("/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    """
    Comprehensive stock analysis with LLM insights
    """
    try:
        logger.info(f"Analyzing stock: {request.symbol}")
        
        # Check cache
        cache_key = f"stock_{request.symbol}_{request.analysis_type}"
        if cache_key in cache:
            cached_data = cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 300:  # 5 min cache
                return cached_data['data']
        
        # Get comprehensive stock data
        stock_data = financial_analyzer.get_comprehensive_analysis(request.symbol)
        
        if "error" in stock_data:
            raise HTTPException(status_code=400, detail=stock_data["error"])
        
        # Generate AI analysis
        provider = LLMProvider.GROQ if request.use_groq else LLMProvider.OPENAI
        ai_analysis = llm_handler.analyze_financial_data(
            financial_data=stock_data,
            analysis_type=request.analysis_type,
            provider=provider
        )
        
        result = {
            "symbol": request.symbol.upper(),
            "timestamp": datetime.utcnow().isoformat(),
            "market_data": stock_data,
            "ai_analysis": ai_analysis,
            "data_sources": ["yfinance", "web_search", ai_analysis["provider_used"]]
        }
        
        # Cache result
        cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing stock {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/portfolio")
async def analyze_portfolio(request: PortfolioRequest):
    """
    Portfolio analysis and optimization recommendations
    """
    try:
        # Analyze portfolio
        portfolio_analysis = portfolio_analyzer.analyze_portfolio(request.holdings)
        
        if request.analyze:
            # Generate AI recommendations
            ai_analysis = llm_handler.analyze_financial_data(
                financial_data=portfolio_analysis,
                analysis_type="comprehensive",
                provider=LLMProvider.GROQ
            )
            portfolio_analysis["ai_recommendations"] = ai_analysis
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": portfolio_analysis
        }
        
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/financial")
async def financial_query(request: FinancialQueryRequest):
    """
    Answer financial queries with RAG, web search augmentation and real-time stock data
    """
    try:
        # Use RAG system if requested
        if request.use_rag:
            try:
                # Get or initialize RAG system
                rag = get_rag_system()
                
                # Detect risk tolerance
                detected_risk = request.risk_tolerance or "moderate"
                query_lower = request.query.lower()
                if any(phrase in query_lower for phrase in ["high risk", "aggressive", "double", "triple", "risky", "volatile", "growth"]):
                    detected_risk = "aggressive"
                elif any(phrase in query_lower for phrase in ["conservative", "safe", "low risk", "stable", "dividend"]):
                    detected_risk = "conservative"
                
                # Query RAG system
                rag_response = rag.query(
                    question=request.query,
                    top_k=5,
                    use_rag=True,
                    risk_tolerance=detected_risk
                )
                
                # Format response
                return {
                    "query": request.query,
                    "response": {
                        "advice": rag_response["answer"],
                        "risk_tolerance": detected_risk,
                        "disclaimer": "This is educational content only. Please consult with a qualified financial advisor for personalized investment advice.",
                        "provider": "openai",
                        "method": rag_response["method"],
                        "processing_time": 0  # Will be calculated by frontend
                    },
                    "rag_sources": rag_response.get("sources", []),
                    "documents_retrieved": rag_response.get("documents_retrieved", 0),
                    "web_search_included": False,  # RAG includes web search data
                    "real_time_data_included": True,  # RAG includes real-time data
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as rag_error:
                logger.error(f"RAG system error, falling back to standard method: {str(rag_error)}")
                # Fall back to standard method if RAG fails
                request.use_rag = False
        
        # Standard method (non-RAG)
        if not request.use_rag:
            context = request.context or {}
            
            # Add web search results if requested
            if request.use_web_search:
                search_results = financial_analyzer.search_financial_news(
                    request.query, 
                    max_results=5
                )
                context["recent_news"] = search_results
            
            # Extract stock symbols from query and add real-time data
            import re
            # Look for known stock symbols or use common tech stocks for tech queries
            common_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
            known_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'CSCO', 'IBM']
            
            # Extract potential stock symbols (2-5 uppercase letters) and filter against known symbols
            potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', request.query.upper())
            stock_symbols = [s for s in potential_symbols if s in known_symbols]
            
            # If no specific symbols mentioned, add data for common tech stocks when query is about tech stocks
            if not stock_symbols and any(word in request.query.lower() for word in ['tech', 'technology', 'stocks', 'buy', 'invest']):
                stock_symbols = common_stocks[:5]
            
            # Get real-time stock data for relevant symbols
            if stock_symbols:
                stock_data = {}
                for symbol in stock_symbols[:10]:  # Limit to 10 stocks
                    try:
                        # Use the same method as the working /stock/{symbol} endpoint
                        stock_info = financial_analyzer.get_stock_info(symbol)
                        if "error" not in stock_info:
                            stock_data[symbol] = stock_info
                            logger.info(f"Successfully fetched data for {symbol}: ${stock_info.get('current_price', 'N/A')}")
                        else:
                            logger.warning(f"Error in stock data for {symbol}: {stock_info.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"Exception fetching data for {symbol}: {str(e)}")
                        continue
                
                if stock_data:
                    context["current_stock_data"] = stock_data
                    logger.info(f"Added stock data for {len(stock_data)} symbols to context")
            
            # Always use OpenAI
            provider = LLMProvider.OPENAI
            
            # Detect risk tolerance from query if not provided
            detected_risk = request.risk_tolerance or "moderate"
            query_lower = request.query.lower()
            if any(phrase in query_lower for phrase in ["high risk", "aggressive", "double", "triple", "risky", "volatile", "growth"]):
                detected_risk = "aggressive"
            elif any(phrase in query_lower for phrase in ["conservative", "safe", "low risk", "stable", "dividend"]):
                detected_risk = "conservative"
            
            # Generate response
            response = llm_handler.generate_investment_advice(
                user_query=request.query,
                context=context,
                risk_tolerance=detected_risk
            )
            
            return {
                "query": request.query,
                "response": response,
                "rag_sources": [],
                "documents_retrieved": 0,
                "web_search_included": request.use_web_search,
                "real_time_data_included": len(stock_data) > 0 if 'stock_data' in locals() else False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error processing financial query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/news")
async def get_market_news(query: str = "stock market", max_results: int = 10):
    """
    Get latest financial news
    """
    try:
        news_results = financial_analyzer.search_financial_news(query, max_results)
        
        return {
            "query": query,
            "count": len(news_results),
            "news": news_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{symbol}")
async def get_stock_info(symbol: str):
    """
    Get basic stock information
    """
    try:
        stock_info = financial_analyzer.get_stock_info(symbol)
        
        if "error" in stock_info:
            raise HTTPException(status_code=404, detail=stock_info["error"])
        
        return stock_info
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{symbol}/ratios")
async def get_stock_ratios(symbol: str):
    """
    Get financial ratios for a stock
    """
    try:
        ratios = financial_analyzer.get_financial_ratios(symbol)
        
        if "error" in ratios:
            raise HTTPException(status_code=404, detail=ratios["error"])
        
        return ratios
        
    except Exception as e:
        logger.error(f"Error fetching ratios for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{symbol}/recommendations")
async def get_analyst_recommendations(symbol: str):
    """
    Get analyst recommendations
    """
    try:
        recommendations = financial_analyzer.get_analyst_recommendations(symbol)
        
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error fetching recommendations for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stock/{symbol}/sentiment")
async def get_market_sentiment(symbol: str):
    """
    Get market sentiment analysis
    """
    try:
        sentiment = financial_analyzer.get_market_sentiment(symbol)
        
        if "error" in sentiment:
            raise HTTPException(status_code=404, detail=sentiment["error"])
        
        return sentiment
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Original endpoints (enhanced)
@app.post("/run_task", response_model=AgentResponse)
async def run_task(task: AgentTask):
    """
    Enhanced task execution with Groq option
    """
    try:
        # Use Groq for faster response if available
        provider = LLMProvider.GROQ if "groq" in llm_handler.get_available_providers() else LLMProvider.OPENAI
        
        system_prompt = task.role + """ 
You are an advanced financial advisor powered by real-time data and AI.
Provide accurate, helpful, and compliant financial education.
Focus on being informative while maintaining regulatory compliance."""
        
        # Generate response
        result = llm_handler.generate_response(
            prompt=task.prompt,
            system_prompt=system_prompt,
            provider=provider,
            model_type="fast" if provider == LLMProvider.GROQ else "balanced",
            temperature=0.7,
            max_tokens=800
        )
        
        return AgentResponse(
            prompt=task.prompt,
            response=result["response"]
        )
        
    except Exception as e:
        logger.error(f"Error in run_task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=Evaluation)
async def evaluate(response: AgentResponse):
    """
    Enhanced evaluation using LLM
    """
    try:
        # Evaluate using LLM handler
        evaluation_result = llm_handler.evaluate_response_quality(
            response=response.response,
            evaluation_criteria=["accuracy", "clarity", "compliance", "helpfulness"]
        )
        
        # Extract scores (with fallbacks)
        accuracy = evaluation_result.get("accuracy", 70)
        clarity = evaluation_result.get("clarity", 70)
        compliance = evaluation_result.get("compliance", 70)
        feedback = evaluation_result.get("summary_feedback", "Evaluation completed")
        
        evaluation = Evaluation(
            prompt=response.prompt,
            response=response.response,
            accuracy=accuracy,
            clarity=clarity,
            compliance=compliance,
            feedback=feedback
        )
        
        # Store in history
        evaluation_history.evaluations.append(evaluation)
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error in evaluate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/certify", response_model=Certification)
async def certify():
    """
    Generate enhanced certification
    """
    if not evaluation_history.evaluations:
        return Certification(
            certified=False,
            issued_at=None,
            attempts=0,
            scores={}
        )
    
    # Calculate averages
    avg_accuracy = evaluation_history.average_accuracy
    avg_clarity = evaluation_history.average_clarity
    avg_compliance = evaluation_history.average_compliance
    overall_avg = evaluation_history.overall_average
    
    # Check if certified (threshold can be configured)
    threshold = 85.0
    certified = (avg_accuracy >= threshold and 
                avg_clarity >= threshold and 
                avg_compliance >= threshold)
    
    certification = Certification(
        agent_name="Financial Intelligence Agent (Enhanced)",
        certified=certified,
        scores={
            "accuracy": round(avg_accuracy, 2),
            "clarity": round(avg_clarity, 2),
            "compliance": round(avg_compliance, 2),
            "overall": round(overall_avg, 2)
        },
        issued_at=datetime.utcnow() if certified else None,
        attempts=len(evaluation_history.evaluations),
        passed_threshold=threshold
    )
    
    return certification


@app.get("/history", response_model=EvaluationHistory)
async def get_history():
    """Get evaluation history"""
    return evaluation_history


@app.delete("/history")
async def clear_history():
    """Clear evaluation history and cache"""
    global evaluation_history, cache
    evaluation_history = EvaluationHistory()
    cache = {}
    return {"message": "History and cache cleared"}


@app.get("/providers")
async def get_providers():
    """Get available LLM providers"""
    return {
        "providers": llm_handler.get_available_providers(),
        "default": "groq" if "groq" in llm_handler.get_available_providers() else "openai"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
