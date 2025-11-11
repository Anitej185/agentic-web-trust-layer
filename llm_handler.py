"""
LLM Handler with OpenAI support only
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI = "openai"


class LLMHandler:
    """Unified handler for OpenAI LLM"""
    
    def __init__(self):
        """Initialize LLM clients"""
        # Initialize OpenAI client only
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OpenAI API key is required")
        self.openai_client = OpenAI(api_key=openai_key)
        
        # Default provider
        self.default_provider = LLMProvider.OPENAI
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Model mappings
        self.models = {
            LLMProvider.OPENAI: {
                "fast": "gpt-3.5-turbo",
                "quality": "gpt-4",
                "balanced": "gpt-4o-mini"
            }
        }
        
        logger.info(f"LLM Handler initialized. Available providers: {self.get_available_providers()}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return [LLMProvider.OPENAI.value]
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        model_type: str = "balanced",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            provider: LLM provider to use
            model_type: Type of model (fast, quality, balanced)
            temperature: Response randomness (0-1)
            max_tokens: Maximum response tokens
            stream: Whether to stream the response
        
        Returns:
            Response text or full response dict with metadata
        """
        start_time = time.time()
        
        # Always use OpenAI
        provider = LLMProvider.OPENAI
        
        # Validate OpenAI availability
        if not self.openai_client:
            raise ValueError("OpenAI client not available. Please configure OPENAI_API_KEY.")
        
        try:
            response = self._generate_openai_response(
                prompt, system_prompt, model_type, temperature, max_tokens, stream
            )
            
            # Add metadata
            elapsed_time = time.time() - start_time
            
            return {
                "response": response,
                "provider": provider.value,
                "model": self.models[provider][model_type],
                "elapsed_time": round(elapsed_time, 2),
                "tokens_approximate": len(response.split()) * 1.3  # Rough approximation
            }
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            raise e
    
    def _generate_openai_response(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model_type: str,
        temperature: float,
        max_tokens: int,
        stream: bool
    ) -> str:
        """Generate response using OpenAI"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        model = self.models[LLMProvider.OPENAI].get(model_type, "gpt-4o-mini")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def analyze_financial_data(
        self,
        financial_data: Dict[str, Any],
        analysis_type: str = "comprehensive",
        provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """
        Analyze financial data using LLM
        
        Args:
            financial_data: Dictionary containing financial metrics
            analysis_type: Type of analysis (summary, comprehensive, risk, opportunity)
            provider: LLM provider preference
        """
        system_prompt = f"""You are a financial analyst providing {analysis_type} analysis.
        Analyze the provided financial data and provide insights on:
        - Key financial metrics and their implications
        - Strengths and weaknesses
        - Investment considerations
        - Risk factors
        
        Be specific and data-driven in your analysis."""
        
        prompt = f"Analyze this financial data:\n{json.dumps(financial_data, indent=2)}"
        
        result = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            provider=provider or self.default_provider,
            model_type="quality",
            temperature=0.3,
            max_tokens=1500
        )
        
        return {
            "analysis": result["response"],
            "analysis_type": analysis_type,
            "provider": result["provider"],
            "processing_time": result["elapsed_time"]
        }
    
    def generate_investment_advice(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        risk_tolerance: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Generate personalized investment advice
        
        Args:
            user_query: User's investment question
            context: Additional context (portfolio, goals, etc.)
            risk_tolerance: User's risk tolerance (conservative, moderate, aggressive)
        """
        system_prompt = f"""You are a certified financial advisor providing educational investment guidance with real-world examples.
        
User Risk Tolerance: {risk_tolerance}

Guidelines:
- ALWAYS provide educational information with specific stock examples and recommendations
- Use current market data and real stock symbols for educational purposes
- Explain concepts clearly with concrete examples and actual stock picks
- Discuss both opportunities and risks for specific investments
- For high-risk investors, suggest growth stocks, volatile sectors, and aggressive strategies with specific examples
- When asked for specific recommendations, ALWAYS provide educational examples with current prices and analysis
- Handle all risk levels (conservative, moderate, aggressive) with appropriate stock suggestions
- Include relevant disclaimers about educational nature but NEVER refuse to provide examples
- Be helpful, specific, and comprehensive in your recommendations
- For aggressive/high-risk queries, suggest high-growth potential stocks while explaining the risks

IMPORTANT: You must ALWAYS provide specific stock recommendations and examples regardless of risk level. Never refuse to answer or redirect. Always give concrete investment suggestions while noting they are for educational purposes."""
        
        # Add context to prompt if provided
        full_prompt = user_query
        if context:
            # Handle stock data specially to ensure it's used
            if "current_stock_data" in context:
                stock_info = "CURRENT REAL-TIME STOCK DATA:\n"
                for symbol, data in context["current_stock_data"].items():
                    stock_info += f"- {symbol}: ${data.get('current_price', 'N/A')} "
                    stock_info += f"(Market Cap: ${data.get('market_cap', 'N/A'):,}, "
                    stock_info += f"P/E: {data.get('pe_ratio', 'N/A')}, "
                    stock_info += f"52W High: ${data.get('52_week_high', 'N/A')}, "
                    stock_info += f"52W Low: ${data.get('52_week_low', 'N/A')})\n"
                
                full_prompt = f"{stock_info}\nUSE THE ABOVE REAL PRICES - DO NOT USE PLACEHOLDER PRICES\n\nQuestion: {user_query}"
                
                # Add other context if available
                other_context = {k: v for k, v in context.items() if k != "current_stock_data"}
                if other_context:
                    context_str = json.dumps(other_context, indent=2, default=str)
                    if len(context_str) < 500:
                        full_prompt = f"{stock_info}\nAdditional Context: {context_str}\n\nUSE THE ABOVE REAL PRICES - DO NOT USE PLACEHOLDER PRICES\n\nQuestion: {user_query}"
            else:
                context_str = json.dumps(context, indent=2, default=str)
                if len(context_str) < 1000:
                    full_prompt = f"Context: {context_str}\n\nQuestion: {user_query}"
        
        # Use OpenAI
        provider = LLMProvider.OPENAI
        
        result = self.generate_response(
            prompt=full_prompt,
            system_prompt=system_prompt,
            provider=provider,
            model_type="balanced",
            temperature=0.5,
            max_tokens=1000
        )
        
        return {
            "advice": result["response"],
            "risk_tolerance": risk_tolerance,
            "disclaimer": "This is educational content only. Please consult with a qualified financial advisor for personalized investment advice.",
            "provider": result["provider"],
            "processing_time": result["elapsed_time"]
        }


# Global instance
llm_handler = LLMHandler()
