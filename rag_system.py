"""
RAG (Retrieval-Augmented Generation) System for Financial Intelligence
Implements vector search, document embedding, and intelligent retrieval
"""

import os
import json
import time
import hashlib
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

# Core dependencies
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

# Vector database and ML dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Web scraping and search
from duckduckgo_search import DDGS
import yfinance as yf

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: Optional[datetime] = None


class FinancialKnowledgeBase:
    """Manages the financial knowledge base with web scraping and data aggregation"""
    
    def __init__(self):
        self.ddgs = DDGS()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_sec_filings(self, ticker: str, filing_type: str = '10-K') -> List[Dict[str, Any]]:
        """Scrape SEC filings for a given ticker"""
        documents = []
        try:
            # Search for recent filings
            search_results = self.ddgs.text(
                f"SEC {ticker} {filing_type} site:sec.gov/Archives/edgar",
                max_results=5
            )
            
            for result in search_results:
                doc = {
                    'source': 'SEC EDGAR',
                    'ticker': ticker,
                    'filing_type': filing_type,
                    'title': result.get('title', ''),
                    'content': result.get('body', ''),
                    'url': result.get('href', ''),
                    'timestamp': datetime.now().isoformat()
                }
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error scraping SEC filings for {ticker}: {str(e)}")
        
        return documents
    
    def scrape_financial_news(self, query: str, max_articles: int = 20) -> List[Dict[str, Any]]:
        """Scrape financial news from multiple sources"""
        documents = []
        
        # Define reliable financial news sources
        sources = [
            "bloomberg.com",
            "reuters.com",
            "wsj.com",
            "cnbc.com",
            "marketwatch.com",
            "seekingalpha.com"
        ]
        
        try:
            for source in sources[:3]:  # Limit to top 3 sources for speed
                search_query = f"{query} site:{source}"
                results = self.ddgs.text(search_query, max_results=max_articles // 3)
                
                for result in results:
                    doc = {
                        'source': source,
                        'type': 'news',
                        'title': result.get('title', ''),
                        'content': result.get('body', ''),
                        'url': result.get('href', ''),
                        'timestamp': datetime.now().isoformat()
                    }
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error scraping financial news: {str(e)}")
        
        return documents
    
    def fetch_market_data(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Fetch comprehensive market data for given tickers"""
        documents = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Format market cap properly
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A' and isinstance(market_cap, (int, float)):
                    market_cap_str = f"${market_cap:,}"
                else:
                    market_cap_str = str(market_cap)
                
                # Create comprehensive market data document
                doc = {
                    'source': 'Yahoo Finance',
                    'type': 'market_data',
                    'ticker': ticker,
                    'content': f"""
                    Company: {info.get('longName', ticker)}
                    Sector: {info.get('sector', 'N/A')}
                    Industry: {info.get('industry', 'N/A')}
                    Current Price: ${info.get('currentPrice', 'N/A')}
                    Market Cap: {market_cap_str}
                    P/E Ratio: {info.get('forwardPE', 'N/A')}
                    52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
                    52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
                    Dividend Yield: {info.get('dividendYield', 'N/A')}
                    Beta: {info.get('beta', 'N/A')}
                    
                    Business Summary: {info.get('longBusinessSummary', 'N/A')}
                    
                    Analyst Recommendation: {info.get('recommendationKey', 'N/A')}
                    Number of Analyst Opinions: {info.get('numberOfAnalystOpinions', 'N/A')}
                    """,
                    'metadata': info,
                    'timestamp': datetime.now().isoformat()
                }
                documents.append(doc)
                
                # Fetch recent price history
                history = stock.history(period="1mo")
                if not history.empty:
                    volume_sum = history['Volume'].sum()
                    price_summary = f"""
                    {ticker} Price History (Last 30 Days):
                    Average Price: ${history['Close'].mean():.2f}
                    Volatility (Std Dev): ${history['Close'].std():.2f}
                    Highest Price: ${history['High'].max():.2f}
                    Lowest Price: ${history['Low'].min():.2f}
                    Total Volume: {volume_sum:,}
                    """
                    
                    documents.append({
                        'source': 'Yahoo Finance',
                        'type': 'price_history',
                        'ticker': ticker,
                        'content': price_summary,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching market data for {ticker}: {str(e)}")
        
        return documents
    
    def scrape_analyst_reports(self, ticker: str) -> List[Dict[str, Any]]:
        """Scrape analyst reports and recommendations"""
        documents = []
        
        try:
            # Search for analyst reports
            search_results = self.ddgs.text(
                f"{ticker} analyst report recommendation price target",
                max_results=10
            )
            
            for result in search_results:
                doc = {
                    'source': 'Web Search',
                    'type': 'analyst_report',
                    'ticker': ticker,
                    'title': result.get('title', ''),
                    'content': result.get('body', ''),
                    'url': result.get('href', ''),
                    'timestamp': datetime.now().isoformat()
                }
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error scraping analyst reports for {ticker}: {str(e)}")
        
        return documents
    
    def build_comprehensive_knowledge_base(self, 
                                          focus_tickers: List[str] = None,
                                          include_news: bool = True,
                                          include_sec: bool = True,
                                          include_analyst: bool = True) -> List[Dict[str, Any]]:
        """Build comprehensive knowledge base from multiple sources"""
        
        # Default popular tickers if none provided
        if not focus_tickers:
            focus_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ']
        
        all_documents = []
        
        logger.info(f"Building knowledge base for {len(focus_tickers)} tickers...")
        
        # Fetch market data for all tickers
        market_docs = self.fetch_market_data(focus_tickers)
        all_documents.extend(market_docs)
        logger.info(f"Fetched {len(market_docs)} market data documents")
        
        # Fetch news if requested
        if include_news:
            news_docs = self.scrape_financial_news("stock market investing analysis 2024", max_articles=20)
            all_documents.extend(news_docs)
            logger.info(f"Fetched {len(news_docs)} news documents")
        
        # Fetch SEC filings if requested
        if include_sec:
            for ticker in focus_tickers[:3]:  # Limit to top 3 for speed
                sec_docs = self.scrape_sec_filings(ticker)
                all_documents.extend(sec_docs)
            logger.info(f"Fetched SEC filings")
        
        # Fetch analyst reports if requested
        if include_analyst:
            for ticker in focus_tickers[:3]:  # Limit to top 3 for speed
                analyst_docs = self.scrape_analyst_reports(ticker)
                all_documents.extend(analyst_docs)
            logger.info(f"Fetched analyst reports")
        
        # Add educational content about investing
        educational_docs = self._get_educational_content()
        all_documents.extend(educational_docs)
        
        logger.info(f"Total documents in knowledge base: {len(all_documents)}")
        
        return all_documents
    
    def _get_educational_content(self) -> List[Dict[str, Any]]:
        """Get educational content about investing"""
        return [
            {
                'source': 'Educational',
                'type': 'guide',
                'title': 'Investment Strategies for 2024',
                'content': """
                Top Investment Strategies for 2024:
                
                1. AI and Technology Stocks: NVDA, MSFT, GOOGL leading the AI revolution
                2. Value Investing: Look for undervalued stocks with P/E < 15
                3. Dividend Aristocrats: JNJ, KO, PG for stable income
                4. Growth Stocks: TSLA, AMZN, META for high returns
                5. ETFs for Diversification: SPY, QQQ, VTI
                
                Risk Management:
                - Diversify across 10-15 stocks minimum
                - Allocate 60% stocks, 30% bonds, 10% alternatives
                - Use stop-loss at 10-15% below purchase price
                - Rebalance quarterly
                
                Best Sectors for 2024:
                - Technology (AI, Cloud, Cybersecurity)
                - Healthcare (Biotech, Medical Devices)
                - Renewable Energy (Solar, Wind, EVs)
                - Financial Services (Digital Banking, Fintech)
                """,
                'timestamp': datetime.now().isoformat()
            },
            {
                'source': 'Educational',
                'type': 'metrics',
                'title': 'Key Financial Metrics Explained',
                'content': """
                Essential Financial Metrics for Stock Analysis:
                
                P/E Ratio (Price-to-Earnings):
                - Below 15: Potentially undervalued (banks, utilities)
                - 15-25: Fair value (most S&P 500 stocks)
                - Above 25: Growth stocks (tech companies)
                - AAPL: ~30, MSFT: ~35, JPM: ~12
                
                Market Cap Categories:
                - Mega Cap: >$200B (AAPL, MSFT, GOOGL)
                - Large Cap: $10B-$200B (NKE, SBUX, GS)
                - Mid Cap: $2B-$10B (growth opportunities)
                - Small Cap: <$2B (higher risk/reward)
                
                Dividend Yields:
                - High Yield: >4% (T, VZ, XOM)
                - Moderate: 2-4% (JNJ, PG, JPM)
                - Growth Focus: <2% (AAPL, MSFT)
                
                Beta (Market Sensitivity):
                - <0.5: Low volatility (utilities)
                - 0.5-1.0: Moderate (consumer staples)
                - 1.0-1.5: Average (most stocks)
                - >1.5: High volatility (TSLA, NVDA)
                """,
                'timestamp': datetime.now().isoformat()
            }
        ]


class RAGSystem:
    """Main RAG system with vector search and retrieval"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_chromadb: bool = False):
        """
        Initialize RAG system
        
        Args:
            embedding_model: Name of the sentence transformer model
            use_chromadb: Whether to use ChromaDB (True) or FAISS (False)
        """
        # Initialize knowledge base
        self.knowledge_base = FinancialKnowledgeBase()
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize embedding model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.use_embeddings = True
        else:
            logger.warning("Sentence transformers not available, using simple text matching")
            self.embedding_model = None
            self.embedding_dim = 0
            self.use_embeddings = False
        
        # Initialize vector database
        self.use_chromadb = use_chromadb and CHROMADB_AVAILABLE
        self.use_faiss = FAISS_AVAILABLE and self.use_embeddings
        
        if self.use_chromadb:
            self._init_chromadb()
        elif self.use_faiss:
            self._init_faiss()
        else:
            # Fallback to simple document storage
            self.document_store = []
            logger.info("Using simple document storage (no vector search)")
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        # Documents storage
        self.documents = []
        
        logger.info(f"RAG System initialized with embedding support: {self.use_embeddings}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB vector database"""
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.create_collection(
                    name="financial_knowledge",
                    metadata={"hnsw:space": "cosine"}
                )
            except:
                self.collection = self.chroma_client.get_collection("financial_knowledge")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            logger.info("Falling back to FAISS")
            self.use_chromadb = False
            self._init_faiss()
    
    def _init_faiss(self):
        """Initialize FAISS vector database"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.document_store = []
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.use_embeddings:
            # Return dummy embedding for text matching fallback
            return np.array([0.0])
            
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Cache embedding (limit cache size)
        if len(self.embedding_cache) < 1000:
            self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        logger.info(f"Adding {len(documents)} documents to storage...")
        
        for i, doc in enumerate(documents):
            try:
                # Create document ID
                doc_id = f"doc_{len(self.documents)}_{i}"
                
                # Generate embedding
                content = doc.get('content', '')
                if not content:
                    continue
                    
                embedding = self.generate_embedding(content)
                
                # Store in vector database
                if self.use_chromadb:
                    self.collection.add(
                        embeddings=[embedding.tolist()],
                        documents=[content],
                        metadatas=[doc],
                        ids=[doc_id]
                    )
                elif self.use_faiss:
                    # FAISS storage
                    self.index.add(np.array([embedding]))
                    self.document_store.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': doc
                    })
                else:
                    # Simple storage fallback
                    self.document_store.append({
                        'id': doc_id,
                        'content': content,
                        'metadata': doc
                    })
                
                # Store document
                self.documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=doc,
                    embedding=embedding,
                    timestamp=datetime.now()
                ))
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error adding document {i}: {str(e)}")
        
        logger.info(f"Successfully added documents. Total documents: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for relevant documents using vector similarity or text matching"""
        results = []
        
        if self.use_chromadb:
            # ChromaDB search
            query_embedding = self.generate_embedding(query)
            search_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, len(self.documents))
            )
            
            if search_results['documents'] and search_results['documents'][0]:
                for i in range(len(search_results['documents'][0])):
                    doc = Document(
                        id=search_results['ids'][0][i],
                        content=search_results['documents'][0][i],
                        metadata=search_results['metadatas'][0][i] if search_results['metadatas'] else {},
                        timestamp=datetime.now()
                    )
                    distance = search_results['distances'][0][i] if search_results['distances'] else 0
                    results.append((doc, 1 - distance))  # Convert distance to similarity
        elif self.use_faiss:
            # FAISS search
            query_embedding = self.generate_embedding(query)
            if len(self.document_store) > 0:
                k = min(top_k, len(self.document_store))
                distances, indices = self.index.search(np.array([query_embedding]), k)
                
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.document_store) and idx >= 0:
                        doc_data = self.document_store[idx]
                        doc = Document(
                            id=doc_data['id'],
                            content=doc_data['content'],
                            metadata=doc_data['metadata'],
                            timestamp=datetime.now()
                        )
                        similarity = 1 / (1 + distances[0][i])  # Convert distance to similarity
                        results.append((doc, similarity))
        else:
            # Simple text matching fallback
            query_lower = query.lower()
            scored_docs = []
            
            for doc_data in self.document_store:
                content_lower = doc_data['content'].lower()
                # Simple keyword matching score
                query_words = query_lower.split()
                matches = sum(1 for word in query_words if word in content_lower)
                score = matches / len(query_words) if query_words else 0
                
                if score > 0:
                    doc = Document(
                        id=doc_data['id'],
                        content=doc_data['content'],
                        metadata=doc_data['metadata'],
                        timestamp=datetime.now()
                    )
                    scored_docs.append((doc, score))
            
            # Sort by score and return top_k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = scored_docs[:top_k]
        
        return results
    
    def generate_augmented_response(self, 
                                   query: str, 
                                   context_docs: List[Document],
                                   model: str = "gpt-4o-mini",
                                   temperature: float = 0.7,
                                   risk_tolerance: str = "moderate") -> str:
        """Generate response using retrieved context"""
        
        # Prepare context from retrieved documents
        context_parts = []
        for doc in context_docs[:5]:  # Limit to top 5 documents
            source = doc.metadata.get('source', 'Unknown')
            doc_type = doc.metadata.get('type', 'general')
            ticker = doc.metadata.get('ticker', '')
            
            context_part = f"[Source: {source}"
            if ticker:
                context_part += f" | Ticker: {ticker}"
            context_part += f" | Type: {doc_type}]\n{doc.content}\n"
            context_parts.append(context_part)
        
        context = "\n---\n".join(context_parts)
        
        # Create augmented prompt
        system_prompt = f"""You are a financial advisor providing educational investment guidance with real-world examples.
        You have access to real-time market data and recent financial information from reliable sources.
        User Risk Tolerance: {risk_tolerance}
        
        Guidelines:
        - Always provide specific, actionable advice with real stock examples and current prices
        - Use the provided context to give accurate, up-to-date information
        - Cite sources when possible (e.g., "According to Yahoo Finance..." or "Based on SEC filings...")
        - Include specific stock recommendations with tickers and current data
        - Explain investment strategies clearly with examples
        - Include relevant disclaimers but never refuse to provide examples or recommendations
        - For educational purposes, provide concrete investment suggestions
        """
        
        user_prompt = f"""Based on the following real-time financial data and information, answer this question:
        
        Question: {query}
        
        Context from Knowledge Base:
        {context}
        
        Please provide a comprehensive answer with specific examples and recommendations based on the data provided.
        Include current prices and metrics where available."""
        
        # Generate response
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 5, use_rag: bool = True, risk_tolerance: str = "moderate") -> Dict[str, Any]:
        """Main query interface for the RAG system"""
        
        if use_rag and len(self.documents) > 0:
            # Search for relevant documents
            search_results = self.search(question, top_k=top_k)
            
            # Extract documents from search results
            relevant_docs = [doc for doc, score in search_results]
            scores = [score for doc, score in search_results]
            
            # Generate augmented response
            answer = self.generate_augmented_response(
                question, 
                relevant_docs, 
                risk_tolerance=risk_tolerance
            )
            
            # Prepare sources for display
            sources = []
            for doc, score in search_results[:3]:
                source_info = {
                    "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "type": doc.metadata.get('type', 'general'),
                    "relevance_score": round(score, 3)
                }
                if 'ticker' in doc.metadata:
                    source_info['ticker'] = doc.metadata['ticker']
                if 'title' in doc.metadata:
                    source_info['title'] = doc.metadata['title']
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "method": "RAG-Enhanced",
                "documents_retrieved": len(search_results)
            }
        else:
            # Fallback to non-RAG response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a helpful financial advisor. User risk tolerance: {risk_tolerance}"},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "answer": response.choices[0].message.content,
                "sources": [],
                "method": "Direct LLM",
                "documents_retrieved": 0
            }
    
    def build_and_index_knowledge_base(self, 
                                      focus_tickers: List[str] = None,
                                      force_rebuild: bool = False) -> int:
        """Build and index the complete knowledge base"""
        
        # Check if we should rebuild
        if not force_rebuild and len(self.documents) > 0:
            logger.info(f"Knowledge base already contains {len(self.documents)} documents")
            return len(self.documents)
        
        # Build knowledge base
        documents = self.knowledge_base.build_comprehensive_knowledge_base(
            focus_tickers=focus_tickers,
            include_news=True,
            include_sec=True,
            include_analyst=True
        )
        
        # Add documents to vector database
        self.add_documents(documents)
        
        return len(self.documents)


# Global RAG instance (will be initialized when needed)
rag_system = None

def get_rag_system() -> RAGSystem:
    """Get or create the global RAG system instance"""
    global rag_system
    if rag_system is None:
        logger.info("Initializing global RAG system...")
        rag_system = RAGSystem(use_chromadb=False)  # Use FAISS for easier setup
        
        # Build initial knowledge base
        doc_count = rag_system.build_and_index_knowledge_base()
        logger.info(f"RAG system ready with {doc_count} documents")
    
    return rag_system
