"""
Financial analysis tools for stock data, market insights, and web search
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Comprehensive financial analysis tools"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.ddgs = DDGS()
        
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Get current price data
            history = ticker.history(period="1d")
            current_price = history['Close'].iloc[-1] if not history.empty else info.get('currentPrice', 0)
            
            # Get key financial metrics
            return {
                "symbol": symbol.upper(),
                "company_name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "current_price": round(current_price, 2),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "forward_pe": info.get('forwardPE', 'N/A'),
                "peg_ratio": info.get('pegRatio', 'N/A'),
                "dividend_yield": info.get('dividendYield', 0),
                "beta": info.get('beta', 'N/A'),
                "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
                "volume": info.get('volume', 0),
                "avg_volume": info.get('averageVolume', 0),
                "profit_margins": info.get('profitMargins', 'N/A'),
                "revenue_growth": info.get('revenueGrowth', 'N/A'),
                "earnings_growth": info.get('earningsGrowth', 'N/A'),
                "return_on_equity": info.get('returnOnEquity', 'N/A'),
                "debt_to_equity": info.get('debtToEquity', 'N/A'),
                "recommendation": info.get('recommendationKey', 'N/A'),
                "target_mean_price": info.get('targetMeanPrice', 'N/A'),
                "analyst_count": info.get('numberOfAnalystOpinions', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
            return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}
    
    def get_stock_history(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol.upper())
            history = ticker.history(period=period)
            return history
        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_analyst_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Get analyst recommendations and ratings"""
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Get recommendations
            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                # Get latest recommendations
                recent_recs = recommendations.tail(10)
                
                # Count recommendation types
                rec_counts = recent_recs['To Grade'].value_counts().to_dict() if 'To Grade' in recent_recs.columns else {}
            else:
                rec_counts = {}
            
            # Get analyst info from info dict
            info = ticker.info
            
            return {
                "symbol": symbol.upper(),
                "recommendation_key": info.get('recommendationKey', 'N/A'),
                "recommendation_mean": info.get('recommendationMean', 'N/A'),
                "target_mean_price": info.get('targetMeanPrice', 'N/A'),
                "target_high_price": info.get('targetHighPrice', 'N/A'),
                "target_low_price": info.get('targetLowPrice', 'N/A'),
                "analyst_count": info.get('numberOfAnalystOpinions', 0),
                "recent_recommendations": rec_counts,
                "price_targets": {
                    "current": info.get('currentPrice', 0),
                    "mean": info.get('targetMeanPrice', 0),
                    "low": info.get('targetLowPrice', 0),
                    "high": info.get('targetHighPrice', 0),
                    "upside_potential": self._calculate_upside(
                        info.get('currentPrice', 0),
                        info.get('targetMeanPrice', 0)
                    )
                }
            }
        except Exception as e:
            logger.error(f"Error fetching analyst recommendations for {symbol}: {str(e)}")
            return {"error": f"Failed to fetch recommendations for {symbol}: {str(e)}"}
    
    def get_financial_ratios(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive financial ratios"""
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            return {
                "symbol": symbol.upper(),
                "valuation_ratios": {
                    "pe_ratio": info.get('trailingPE', 'N/A'),
                    "forward_pe": info.get('forwardPE', 'N/A'),
                    "peg_ratio": info.get('pegRatio', 'N/A'),
                    "price_to_book": info.get('priceToBook', 'N/A'),
                    "price_to_sales": info.get('priceToSalesTrailing12Months', 'N/A'),
                    "enterprise_value": info.get('enterpriseValue', 'N/A'),
                    "ev_to_revenue": info.get('enterpriseToRevenue', 'N/A'),
                    "ev_to_ebitda": info.get('enterpriseToEbitda', 'N/A')
                },
                "profitability_ratios": {
                    "profit_margins": info.get('profitMargins', 'N/A'),
                    "operating_margins": info.get('operatingMargins', 'N/A'),
                    "gross_margins": info.get('grossMargins', 'N/A'),
                    "ebitda_margins": info.get('ebitdaMargins', 'N/A'),
                    "return_on_assets": info.get('returnOnAssets', 'N/A'),
                    "return_on_equity": info.get('returnOnEquity', 'N/A')
                },
                "liquidity_ratios": {
                    "current_ratio": info.get('currentRatio', 'N/A'),
                    "quick_ratio": info.get('quickRatio', 'N/A'),
                    "cash": info.get('totalCash', 'N/A'),
                    "cash_per_share": info.get('totalCashPerShare', 'N/A'),
                    "debt": info.get('totalDebt', 'N/A'),
                    "debt_to_equity": info.get('debtToEquity', 'N/A')
                },
                "growth_metrics": {
                    "revenue_growth": info.get('revenueGrowth', 'N/A'),
                    "earnings_growth": info.get('earningsGrowth', 'N/A'),
                    "earnings_quarterly_growth": info.get('earningsQuarterlyGrowth', 'N/A'),
                    "revenue_per_share": info.get('revenuePerShare', 'N/A'),
                    "book_value": info.get('bookValue', 'N/A')
                }
            }
        except Exception as e:
            logger.error(f"Error fetching financial ratios for {symbol}: {str(e)}")
            return {"error": f"Failed to fetch ratios for {symbol}: {str(e)}"}
    
    def search_financial_news(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Search for financial news using DuckDuckGo"""
        try:
            results = []
            search_results = self.ddgs.text(
                f"{query} stock market financial news",
                max_results=max_results
            )
            
            for result in search_results:
                results.append({
                    "title": result.get('title', ''),
                    "snippet": result.get('body', ''),
                    "url": result.get('href', ''),
                    "source": result.get('href', '').split('/')[2] if result.get('href') else 'Unknown'
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching news for {query}: {str(e)}")
            return []
    
    def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get market sentiment and social indicators"""
        try:
            # Search for recent news and sentiment
            news_results = self.search_financial_news(f"{symbol} stock", max_results=20)
            
            # Analyze sentiment from news titles (simplified sentiment analysis)
            positive_words = ['buy', 'bullish', 'upgrade', 'growth', 'profit', 'beats', 'surge', 'gain', 'rally']
            negative_words = ['sell', 'bearish', 'downgrade', 'loss', 'miss', 'decline', 'fall', 'drop', 'crash']
            
            sentiment_score = 0
            for news in news_results:
                title_lower = news['title'].lower()
                sentiment_score += sum(1 for word in positive_words if word in title_lower)
                sentiment_score -= sum(1 for word in negative_words if word in title_lower)
            
            # Normalize sentiment score
            sentiment_label = 'Neutral'
            if sentiment_score > 3:
                sentiment_label = 'Bullish'
            elif sentiment_score < -3:
                sentiment_label = 'Bearish'
            
            return {
                "symbol": symbol.upper(),
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "news_count": len(news_results),
                "recent_headlines": [news['title'] for news in news_results[:5]],
                "news_sources": list(set([news['source'] for news in news_results]))
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {"error": f"Failed to analyze sentiment for {symbol}: {str(e)}"}
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock analysis combining all data sources"""
        try:
            # Gather all data in parallel for efficiency
            stock_info = self.get_stock_info(symbol)
            analyst_recs = self.get_analyst_recommendations(symbol)
            ratios = self.get_financial_ratios(symbol)
            sentiment = self.get_market_sentiment(symbol)
            
            # Get recent price history for trend analysis
            history = self.get_stock_history(symbol, period="1mo")
            
            trend = "Neutral"
            if not history.empty and len(history) > 1:
                price_change = ((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0]) * 100
                if price_change > 5:
                    trend = "Uptrend"
                elif price_change < -5:
                    trend = "Downtrend"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol.upper(),
                "overview": stock_info,
                "analyst_recommendations": analyst_recs,
                "financial_ratios": ratios,
                "market_sentiment": sentiment,
                "trend_analysis": {
                    "trend": trend,
                    "1_month_return": round(price_change, 2) if not history.empty else "N/A",
                    "volatility": round(history['Close'].std(), 2) if not history.empty else "N/A"
                },
                "risk_assessment": self._calculate_risk_score(stock_info, ratios, sentiment)
            }
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {str(e)}")
            return {"error": f"Failed to complete analysis for {symbol}: {str(e)}"}
    
    def _calculate_upside(self, current_price: float, target_price: float) -> str:
        """Calculate upside potential percentage"""
        try:
            if current_price and target_price and current_price > 0:
                upside = ((target_price - current_price) / current_price) * 100
                return f"{round(upside, 2)}%"
            return "N/A"
        except:
            return "N/A"
    
    def _calculate_risk_score(self, stock_info: Dict, ratios: Dict, sentiment: Dict) -> Dict[str, Any]:
        """Calculate risk score based on multiple factors"""
        risk_factors = []
        risk_score = 50  # Start with neutral score
        
        # Check P/E ratio
        pe = stock_info.get('pe_ratio', 'N/A')
        if pe != 'N/A' and pe > 30:
            risk_factors.append("High P/E ratio")
            risk_score += 10
        elif pe != 'N/A' and pe < 10:
            risk_factors.append("Low P/E ratio (potential value or risk)")
            risk_score += 5
        
        # Check debt to equity
        if 'liquidity_ratios' in ratios:
            debt_to_equity = ratios['liquidity_ratios'].get('debt_to_equity', 'N/A')
            if debt_to_equity != 'N/A' and debt_to_equity > 2:
                risk_factors.append("High debt levels")
                risk_score += 15
        
        # Check sentiment
        if sentiment.get('sentiment_label') == 'Bearish':
            risk_factors.append("Negative market sentiment")
            risk_score += 10
        elif sentiment.get('sentiment_label') == 'Bullish':
            risk_score -= 10
        
        # Determine risk level
        risk_level = "Low"
        if risk_score > 70:
            risk_level = "High"
        elif risk_score > 50:
            risk_level = "Medium"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors
        }


class PortfolioAnalyzer:
    """Portfolio analysis and optimization tools"""
    
    def __init__(self):
        self.financial_analyzer = FinancialAnalyzer()
    
    def analyze_portfolio(self, holdings: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze a portfolio of holdings
        
        Args:
            holdings: List of dicts with 'symbol' and 'shares' keys
        """
        portfolio_data = []
        total_value = 0
        
        for holding in holdings:
            symbol = holding['symbol']
            shares = holding['shares']
            
            stock_info = self.financial_analyzer.get_stock_info(symbol)
            if 'error' not in stock_info:
                current_price = stock_info['current_price']
                position_value = current_price * shares
                total_value += position_value
                
                portfolio_data.append({
                    'symbol': symbol,
                    'shares': shares,
                    'current_price': current_price,
                    'position_value': position_value,
                    'pe_ratio': stock_info.get('pe_ratio', 'N/A'),
                    'dividend_yield': stock_info.get('dividend_yield', 0),
                    'sector': stock_info.get('sector', 'Unknown')
                })
        
        # Calculate portfolio metrics
        if total_value > 0:
            for position in portfolio_data:
                position['weight'] = (position['position_value'] / total_value) * 100
        
        # Sector allocation
        sector_allocation = {}
        for position in portfolio_data:
            sector = position['sector']
            if sector not in sector_allocation:
                sector_allocation[sector] = 0
            sector_allocation[sector] += position.get('weight', 0)
        
        return {
            "total_value": round(total_value, 2),
            "positions": portfolio_data,
            "sector_allocation": sector_allocation,
            "diversification_score": self._calculate_diversification(portfolio_data),
            "recommendations": self._generate_portfolio_recommendations(portfolio_data, sector_allocation)
        }
    
    def _calculate_diversification(self, portfolio_data: List[Dict]) -> Dict[str, Any]:
        """Calculate portfolio diversification metrics"""
        weights = [p.get('weight', 0) for p in portfolio_data]
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum([w**2 for w in weights])
        
        # Determine diversification level
        if hhi < 1000:
            level = "Well Diversified"
        elif hhi < 2500:
            level = "Moderately Diversified"
        else:
            level = "Concentrated"
        
        return {
            "hhi": round(hhi, 2),
            "level": level,
            "position_count": len(portfolio_data)
        }
    
    def _generate_portfolio_recommendations(self, portfolio_data: List[Dict], sector_allocation: Dict) -> List[str]:
        """Generate portfolio recommendations"""
        recommendations = []
        
        # Check for over-concentration
        max_weight = max([p.get('weight', 0) for p in portfolio_data]) if portfolio_data else 0
        if max_weight > 25:
            recommendations.append(f"Consider reducing concentration - highest position is {max_weight:.1f}% of portfolio")
        
        # Check sector allocation
        for sector, weight in sector_allocation.items():
            if weight > 30:
                recommendations.append(f"High exposure to {sector} sector ({weight:.1f}%) - consider diversifying")
        
        # Check number of holdings
        if len(portfolio_data) < 5:
            recommendations.append("Consider adding more positions for better diversification")
        elif len(portfolio_data) > 20:
            recommendations.append("Large number of holdings may make portfolio difficult to manage")
        
        if not recommendations:
            recommendations.append("Portfolio appears well-balanced")
        
        return recommendations


# Singleton instances for efficient resource usage
financial_analyzer = FinancialAnalyzer()
portfolio_analyzer = PortfolioAnalyzer()
