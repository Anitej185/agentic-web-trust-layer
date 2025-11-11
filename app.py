"""
Enhanced Streamlit frontend with Groq LLM and Financial Analysis
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Financial Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5px 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000"

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Stock Analysis"


def check_backend():
    """Check if backend is running and get system status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def format_number(num):
    """Format large numbers for display"""
    if num is None or num == 'N/A':
        return 'N/A'
    if isinstance(num, str):
        return num
    if num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


def create_stock_chart(symbol: str, period: str = "1mo"):
    """Create interactive stock price chart"""
    try:
        # This would normally fetch from yfinance through the backend
        # For demo purposes, creating sample data
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=30)
        prices = np.random.randn(30).cumsum() + 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None


def main():
    """Main Streamlit app"""
    
    # Header with gradient
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0;">🚀 Financial Intelligence System</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0; margin-top: 0.5rem;">
                Powered by Groq LLM, Real-time Data & Web Search
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check backend status
    backend_status = check_backend()
    
    if not backend_status:
        st.error("⚠️ Backend API is not running. Please start it with: `python backend_enhanced.py`")
        st.stop()
    
    # Display system status in sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        if backend_status:
            services = backend_status.get('services', {})
            
            # Create status indicators
            status_cols = st.columns(2)
            with status_cols[0]:
                groq_status = "🟢 Active" if services.get('groq') else "🔴 Inactive"
                st.metric("Groq LLM", groq_status)
                
                financial_status = "🟢 Active" if services.get('financial_data') else "🔴 Inactive"
                st.metric("Market Data", financial_status)
            
            with status_cols[1]:
                openai_status = "🟢 Active" if services.get('openai') else "🔴 Inactive"
                st.metric("OpenAI", openai_status)
                
                search_status = "🟢 Active" if services.get('web_search') else "🔴 Inactive"
                st.metric("Web Search", search_status)
        
        st.markdown("---")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            requests.delete(f"{API_URL}/history")
            st.session_state.stock_data = None
            st.session_state.analysis_history = []
            st.success("Cache cleared!")
            st.rerun()
        
        if st.button("📊 Market Overview", use_container_width=True):
            st.session_state.current_tab = "Market Overview"
            st.rerun()
    
    # Main navigation tabs
    tabs = st.tabs([
        "📈 Stock Analysis",
        "💼 Portfolio Manager", 
        "🔍 Financial Q&A",
        "📰 Market News",
        "🎯 AI Assistant"
    ])
    
    # Tab 1: Stock Analysis
    with tabs[0]:
        st.header("Stock Analysis & Intelligence")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            symbol = st.text_input(
                "Enter Stock Symbol",
                placeholder="e.g., AAPL, GOOGL, MSFT",
                key="stock_symbol"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["comprehensive", "summary", "risk", "opportunity"],
                key="analysis_type"
            )
        
        with col3:
            use_groq = st.checkbox("⚡ Use Groq (Faster)", value=True, key="use_groq")
        
        if st.button("🔍 Analyze Stock", type="primary", use_container_width=True):
            if symbol:
                with st.spinner(f"Analyzing {symbol.upper()}... {'(Using Groq for ultra-fast response)' if use_groq else ''}"):
                    try:
                        # Call enhanced backend
                        response = requests.post(
                            f"{API_URL}/analyze/stock",
                            json={
                                "symbol": symbol,
                                "analysis_type": analysis_type,
                                "use_groq": use_groq
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.stock_data = data
                            
                            # Display results
                            st.success(f"✅ Analysis complete for {symbol.upper()}")
                            
                            # Key metrics
                            if 'market_data' in data and 'overview' in data['market_data']:
                                overview = data['market_data']['overview']
                                
                                st.subheader("📊 Key Metrics")
                                metrics_cols = st.columns(4)
                                
                                with metrics_cols[0]:
                                    st.metric(
                                        "Current Price",
                                        format_number(overview.get('current_price', 'N/A')),
                                        delta=None
                                    )
                                
                                with metrics_cols[1]:
                                    st.metric(
                                        "P/E Ratio",
                                        overview.get('pe_ratio', 'N/A')
                                    )
                                
                                with metrics_cols[2]:
                                    st.metric(
                                        "Market Cap",
                                        format_number(overview.get('market_cap', 'N/A'))
                                    )
                                
                                with metrics_cols[3]:
                                    recommendation = overview.get('recommendation', 'N/A')
                                    st.metric(
                                        "Recommendation",
                                        recommendation.upper() if recommendation != 'N/A' else 'N/A'
                                    )
                            
                            # AI Analysis
                            if 'ai_analysis' in data:
                                st.subheader("🤖 AI Analysis")
                                
                                # Display provider and processing time
                                ai_info = data['ai_analysis']
                                info_cols = st.columns(3)
                                with info_cols[0]:
                                    st.info(f"Provider: {ai_info.get('provider_used', 'Unknown').upper()}")
                                with info_cols[1]:
                                    st.info(f"Processing: {ai_info.get('processing_time', 'N/A')}s")
                                with info_cols[2]:
                                    st.info(f"Analysis: {ai_info.get('analysis_type', 'N/A').title()}")
                                
                                # Display analysis
                                st.markdown(ai_info.get('analysis', 'No analysis available'))
                            
                            # Financial Ratios
                            if 'market_data' in data and 'financial_ratios' in data['market_data']:
                                ratios = data['market_data']['financial_ratios']
                                
                                st.subheader("📊 Financial Ratios")
                                
                                ratio_tabs = st.tabs(["Valuation", "Profitability", "Liquidity", "Growth"])
                                
                                with ratio_tabs[0]:
                                    if 'valuation_ratios' in ratios:
                                        val_ratios = ratios['valuation_ratios']
                                        val_df = pd.DataFrame([val_ratios]).T
                                        val_df.columns = ['Value']
                                        st.dataframe(val_df, use_container_width=True)
                                
                                with ratio_tabs[1]:
                                    if 'profitability_ratios' in ratios:
                                        prof_ratios = ratios['profitability_ratios']
                                        prof_df = pd.DataFrame([prof_ratios]).T
                                        prof_df.columns = ['Value']
                                        st.dataframe(prof_df, use_container_width=True)
                                
                                with ratio_tabs[2]:
                                    if 'liquidity_ratios' in ratios:
                                        liq_ratios = ratios['liquidity_ratios']
                                        liq_df = pd.DataFrame([liq_ratios]).T
                                        liq_df.columns = ['Value']
                                        st.dataframe(liq_df, use_container_width=True)
                                
                                with ratio_tabs[3]:
                                    if 'growth_metrics' in ratios:
                                        growth_metrics = ratios['growth_metrics']
                                        growth_df = pd.DataFrame([growth_metrics]).T
                                        growth_df.columns = ['Value']
                                        st.dataframe(growth_df, use_container_width=True)
                            
                            # Market Sentiment
                            if 'market_data' in data and 'market_sentiment' in data['market_data']:
                                sentiment = data['market_data']['market_sentiment']
                                
                                st.subheader("🎭 Market Sentiment")
                                
                                sent_cols = st.columns(3)
                                with sent_cols[0]:
                                    sentiment_label = sentiment.get('sentiment_label', 'Neutral')
                                    sentiment_color = {
                                        'Bullish': 'green',
                                        'Bearish': 'red',
                                        'Neutral': 'gray'
                                    }.get(sentiment_label, 'gray')
                                    
                                    st.markdown(f"""
                                        <div style="text-align: center; padding: 1rem; 
                                                    background-color: {sentiment_color}20; 
                                                    border-radius: 10px;">
                                            <h3 style="color: {sentiment_color};">{sentiment_label}</h3>
                                            <p>Sentiment Score: {sentiment.get('sentiment_score', 0)}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with sent_cols[1]:
                                    st.metric("News Articles", sentiment.get('news_count', 0))
                                
                                with sent_cols[2]:
                                    st.metric("Data Sources", len(sentiment.get('news_sources', [])))
                                
                                # Recent headlines
                                if 'recent_headlines' in sentiment:
                                    st.markdown("**Recent Headlines:**")
                                    for headline in sentiment['recent_headlines'][:5]:
                                        st.write(f"• {headline}")
                            
                            # Stock chart
                            chart = create_stock_chart(symbol)
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                            
                        else:
                            st.error(f"Error: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error analyzing stock: {str(e)}")
            else:
                st.warning("Please enter a stock symbol")
    
    # Tab 2: Portfolio Manager
    with tabs[1]:
        st.header("Portfolio Analysis & Optimization")
        
        # Portfolio input
        st.subheader("Add Holdings")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            portfolio_symbol = st.text_input("Symbol", placeholder="AAPL", key="portfolio_symbol")
        with col2:
            shares = st.number_input("Shares", min_value=0.0, step=1.0, key="portfolio_shares")
        with col3:
            if st.button("➕ Add", use_container_width=True):
                if portfolio_symbol and shares > 0:
                    st.session_state.portfolio.append({
                        "symbol": portfolio_symbol.upper(),
                        "shares": shares
                    })
                    st.success(f"Added {shares} shares of {portfolio_symbol.upper()}")
                    st.rerun()
        
        # Display current portfolio
        if st.session_state.portfolio:
            st.subheader("Current Holdings")
            portfolio_df = pd.DataFrame(st.session_state.portfolio)
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Analyze portfolio button
            if st.button("🔍 Analyze Portfolio", type="primary", use_container_width=True):
                with st.spinner("Analyzing portfolio..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/analyze/portfolio",
                            json={
                                "holdings": st.session_state.portfolio,
                                "analyze": True
                            }
                        )
                        
                        if response.status_code == 200:
                            portfolio_analysis = response.json()['analysis']
                            
                            # Display portfolio value
                            st.metric(
                                "Total Portfolio Value",
                                format_number(portfolio_analysis.get('total_value', 0))
                            )
                            
                            # Position details
                            if 'positions' in portfolio_analysis:
                                st.subheader("Position Details")
                                positions_df = pd.DataFrame(portfolio_analysis['positions'])
                                st.dataframe(positions_df, use_container_width=True)
                            
                            # Sector allocation pie chart
                            if 'sector_allocation' in portfolio_analysis:
                                st.subheader("Sector Allocation")
                                sectors = list(portfolio_analysis['sector_allocation'].keys())
                                values = list(portfolio_analysis['sector_allocation'].values())
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=sectors,
                                    values=values,
                                    hole=0.3
                                )])
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Diversification score
                            if 'diversification_score' in portfolio_analysis:
                                div_score = portfolio_analysis['diversification_score']
                                st.subheader("Diversification Analysis")
                                
                                div_cols = st.columns(2)
                                with div_cols[0]:
                                    st.metric("Diversification Level", div_score.get('level', 'N/A'))
                                with div_cols[1]:
                                    st.metric("HHI Score", div_score.get('hhi', 'N/A'))
                            
                            # AI Recommendations
                            if 'ai_recommendations' in portfolio_analysis:
                                st.subheader("🤖 AI Recommendations")
                                ai_rec = portfolio_analysis['ai_recommendations']
                                st.info(f"Analysis by: {ai_rec.get('provider_used', 'Unknown').upper()}")
                                st.markdown(ai_rec.get('analysis', 'No recommendations available'))
                        
                    except Exception as e:
                        st.error(f"Error analyzing portfolio: {str(e)}")
            
            # Clear portfolio button
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.portfolio = []
                st.success("Portfolio cleared")
                st.rerun()
        else:
            st.info("No holdings in portfolio. Add stocks above to get started.")
    
    # Tab 3: Financial Q&A
    with tabs[2]:
        st.header("🤖 Financial Q&A with RAG System")
        
        st.markdown("""
        Ask any financial question and get AI-powered answers enhanced with:
        - 📚 **RAG System**: Retrieval from verified financial knowledge base
        - 🔍 **Web Search**: Real-time news and market data
        - 📊 **Source Attribution**: See where information comes from
        """)
        
        # Question input
        question = st.text_area(
            "Your Financial Question",
            placeholder="What are the best dividend stocks for 2024?",
            height=100,
            key="financial_question"
        )
        
        # Options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["conservative", "moderate", "aggressive"],
                index=1,
                key="risk_tolerance"
            )
        with col2:
            use_rag = st.checkbox("🚀 Use RAG System", value=True, key="use_rag", 
                                  help="Uses advanced retrieval system with verified financial sources")
        with col3:
            use_web = st.checkbox("🌐 Include Web Search", value=True, key="use_web")
        with col4:
            provider = st.selectbox(
                "AI Provider",
                ["openai"],
                key="qa_provider"
            )
        
        if st.button("💡 Get Answer", type="primary", use_container_width=True):
            if question:
                with st.spinner("🔍 Searching knowledge base and generating answer..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/query/financial",
                            json={
                                "query": question,
                                "use_web_search": use_web,
                                "use_rag": use_rag,
                                "risk_tolerance": risk_tolerance,
                                "provider": provider
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display response info
                            st.success("✅ Answer generated successfully!")
                            
                            resp_data = result['response']
                            
                            # Info bar with enhanced metrics
                            info_cols = st.columns(4)
                            with info_cols[0]:
                                method = resp_data.get('method', 'Standard')
                                method_icon = "🚀" if "RAG" in method else "💬"
                                st.info(f"{method_icon} Method: {method}")
                            with info_cols[1]:
                                st.info(f"🤖 Provider: {resp_data.get('provider', 'Unknown').upper()}")
                            with info_cols[2]:
                                docs_count = result.get('documents_retrieved', 0)
                                st.info(f"📚 Sources: {docs_count}")
                            with info_cols[3]:
                                st.info(f"⚖️ Risk: {risk_tolerance.title()}")
                            
                            # Display answer
                            st.markdown("### 💡 Answer")
                            st.markdown(resp_data.get('advice', 'No answer available'))
                            
                            # Display RAG sources if available
                            if 'rag_sources' in result and result['rag_sources']:
                                st.markdown("### 📚 Knowledge Base Sources")
                                
                                for idx, source in enumerate(result['rag_sources'], 1):
                                    with st.expander(f"Source {idx}: {source.get('source', 'Unknown')} - {source.get('type', 'general').title()}"):
                                        if 'ticker' in source:
                                            st.markdown(f"**Ticker:** {source['ticker']}")
                                        if 'title' in source:
                                            st.markdown(f"**Title:** {source['title']}")
                                        st.markdown(f"**Relevance Score:** {source.get('relevance_score', 0):.3f}")
                                        st.markdown("**Content Preview:**")
                                        st.text(source.get('content_preview', 'No content available'))
                            
                            # Disclaimer
                            st.warning(resp_data.get('disclaimer', 'This is educational content only.'))
                        
                    except Exception as e:
                        st.error(f"Error getting answer: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    # Tab 4: Market News
    with tabs[3]:
        st.header("Latest Financial News")
        
        # News search
        col1, col2 = st.columns([3, 1])
        with col1:
            news_query = st.text_input(
                "Search News",
                placeholder="e.g., Tesla earnings, Fed rate decision",
                value="stock market",
                key="news_query"
            )
        with col2:
            max_results = st.number_input(
                "Max Results",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="max_results"
            )
        
        if st.button("📰 Search News", type="primary", use_container_width=True):
            with st.spinner("Fetching latest news..."):
                try:
                    response = requests.get(
                        f"{API_URL}/market/news",
                        params={"query": news_query, "max_results": max_results}
                    )
                    
                    if response.status_code == 200:
                        news_data = response.json()
                        
                        st.success(f"Found {news_data['count']} articles")
                        
                        # Display news articles
                        for article in news_data['news']:
                            with st.expander(article['title']):
                                st.markdown(f"**Source:** {article['source']}")
                                st.markdown(article['snippet'])
                                st.markdown(f"[Read more]({article['url']})")
                    
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")
    
    # Tab 5: AI Assistant (Original functionality)
    with tabs[4]:
        st.header("AI Financial Assistant")
        
        # Predefined questions
        st.subheader("Quick Examples")
        example_questions = [
            "I have $500 to invest and no stock background, give me the exact stocks to invest in",
            "What are the top 5 tech stocks to buy right now with current prices?",
            "Give me specific dividend stocks for a $10K retirement portfolio",
            "What are the best growth stocks under $50 per share today?",
            "Recommend 3 specific ETFs for a beginner with current performance data"
        ]
        
        selected_example = st.selectbox(
            "Choose an example or enter your own:",
            [""] + example_questions,
            key="assistant_example"
        )
        
        # Task input
        task_prompt = st.text_area(
            "Enter your financial scenario:",
            value=selected_example if selected_example else "",
            height=100,
            placeholder="Ask me anything about finance and investing...",
            key="assistant_prompt"
        )
        
        # Provider selection (OpenAI only)
        assistant_provider = "🎯 OpenAI (High quality)"
        st.info("🤖 Using OpenAI for all responses")
        
        # Include web search option
        include_web_search = st.checkbox("🔍 Include Web Search", value=True, key="include_web_search")
        
        # Run task button
        if st.button("🤖 Get Response", type="primary", use_container_width=True):
            if task_prompt:
                with st.spinner(f"AI is thinking... {assistant_provider.split()[0]}"):
                    try:
                        # Determine provider
                        use_groq = "Groq" in assistant_provider
                        
                        # Use enhanced financial query endpoint
                        response = requests.post(
                            f"{API_URL}/query/financial",
                            json={
                                "query": task_prompt,
                                "use_web_search": include_web_search,
                                "context": {
                                    "provider": "groq" if use_groq else "openai",
                                    "risk_tolerance": "moderate"
                                }
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            st.success("✅ Enhanced response generated with real-time data!")
                            
                            # Display response
                            st.markdown("### 📝 AI Financial Advisor Response")
                            if 'response' in data and 'advice' in data['response']:
                                st.markdown(data['response']['advice'])
                                
                                # Show additional info
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.info(f"🤖 Provider: {data['response'].get('provider', 'N/A').upper()}")
                                with col2:
                                    st.info(f"⏱️ Processing: {data['response'].get('processing_time', 0):.1f}s")
                                with col3:
                                    search_status = "✅ Included" if data.get('web_search_included') else "❌ Not used"
                                    st.info(f"🔍 Web Search: {search_status}")
                                with col4:
                                    risk_level = data['response'].get('risk_tolerance', 'moderate').title()
                                    risk_emoji = "🔥" if risk_level == "Aggressive" else "🛡️" if risk_level == "Conservative" else "⚖️"
                                    st.info(f"{risk_emoji} Risk: {risk_level}")
                                
                                # Show disclaimer
                                st.warning(f"⚠️ {data['response'].get('disclaimer', 'Educational content only.')}")
                            else:
                                # Fallback for different response format
                                st.markdown(data.get('response', 'No response available'))
                            
                            # Evaluate button
                            if st.button("📊 Evaluate Response"):
                                with st.spinner("Evaluating..."):
                                    eval_response = requests.post(
                                        f"{API_URL}/evaluate",
                                        json=data
                                    )
                                    
                                    if eval_response.status_code == 200:
                                        eval_data = eval_response.json()
                                        
                                        st.markdown("### 📈 Evaluation Results")
                                        
                                        eval_cols = st.columns(4)
                                        with eval_cols[0]:
                                            st.metric("Accuracy", f"{eval_data['accuracy']}/100")
                                        with eval_cols[1]:
                                            st.metric("Clarity", f"{eval_data['clarity']}/100")
                                        with eval_cols[2]:
                                            st.metric("Compliance", f"{eval_data['compliance']}/100")
                                        with eval_cols[3]:
                                            avg_score = eval_data.get('average_score', 0)
                                            st.metric("Average", f"{avg_score:.1f}/100")
                                        
                                        st.markdown("**Feedback:**")
                                        st.info(eval_data['feedback'])
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a prompt")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Financial Intelligence System v2.0 | Powered by Groq LLM, YFinance & DuckDuckGo</p>
            <p style="font-size: 0.9rem;">⚠️ Educational purposes only. Not financial advice.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
