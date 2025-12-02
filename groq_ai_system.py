import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from config import Config
from supabase_manager import SupabaseManager

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class TradingSignal(BaseModel):
    signal_type: str = Field(description="BUY_CALL, BUY_PUT, SELL_CALL, SELL_PUT, or HOLD")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    rationale: str = Field(description="Detailed reasoning for the signal")
    key_risk_factors: List[str] = Field(description="List of key risk factors")
    price_target: float = Field(description="Specific price target")
    stop_loss: float = Field(description="Specific stop loss")
    timeframe_hours: int = Field(description="Expected trade duration in hours")
    position_size_percent: float = Field(description="% of portfolio to allocate", ge=0.1, le=5)
    entry_strategy: str = Field(description="Specific entry strategy")
    exit_strategy: str = Field(description="Specific exit strategy")
    monitoring_parameters: List[str] = Field(description="List of things to monitor")
    alternate_scenarios: List[str] = Field(description="Possible alternate outcomes")

class MarketAnalysis(BaseModel):
    condition: str = Field(description="Current market condition")
    bias: str = Field(description="Trading bias: bullish, bearish, or neutral")
    risk_level: str = Field(description="Risk level: low, medium, high")
    key_levels: Dict[str, float] = Field(description="Key support and resistance levels")
    summary: str = Field(description="Summary analysis")

class QueryAnalysis(BaseModel):
    main_intent: str = Field(description="Primary intent of the query")
    required_metrics: List[str] = Field(description="List of trading metrics mentioned")
    timeframe: str = Field(description="Timeframe mentioned")
    data_points: List[str] = Field(description="Specific data points needed")
    symbol_inferred: Optional[str] = Field(description="Inferred trading symbol")
    expiry_context: Optional[str] = Field(description="Expiry date context")

class GroqAIAnalyzer:
    """AI analysis system using Groq with Llama models"""
    
    def __init__(self, groq_api_key: str = None, 
                 supabase_manager: SupabaseManager = None,
                 model_name: str = None):
        
        self.groq_api_key = groq_api_key or Config.GROQ_API_KEY
        if not self.groq_api_key:
            raise ValueError("Groq API Key is required")
        
        self.supabase = supabase_manager
        self.model_name = model_name or Config.GROQ_MODEL
        
        # Initialize Groq client
        self.client = Groq(api_key=self.groq_api_key)
        
        # Initialize LangChain with Groq
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            temperature=Config.AI_TEMPERATURE,
            max_tokens=Config.AI_MAX_TOKENS
        )
        
        # Available models for different tasks
        self.model_map = Config.get_groq_models()
        
        # Output parsers
        self.signal_parser = PydanticOutputParser(pydantic_object=TradingSignal)
        self.analysis_parser = PydanticOutputParser(pydantic_object=MarketAnalysis)
        self.query_parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
        
        # Initialize usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'errors': 0,
            'models_used': {}
        }
        
        logger.info(f"Groq AI Analyzer initialized with model: {self.model_name}")
    
    async def query_data(self, question: str, symbol: str = None) -> Dict:
        """Query data using natural language with Groq"""
        
        self.usage_stats['total_requests'] += 1
        
        try:
            # Step 1: Analyze the query
            query_analysis = await self._analyze_query_with_groq(question)
            
            # Step 2: Fetch relevant data
            data_context = await self._fetch_relevant_data(query_analysis, symbol)
            
            # Step 3: Generate response
            response = await self._generate_response_with_groq(question, data_context)
            
            return {
                'question': question,
                'query_analysis': query_analysis.dict() if hasattr(query_analysis, 'dict') else query_analysis,
                'data_context_keys': list(data_context.keys()),
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }
            
        except Exception as e:
            self.usage_stats['errors'] += 1
            logger.error(f"Error in query_data: {e}")
            return {
                'question': question,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_query_with_groq(self, question: str) -> QueryAnalysis:
        """Analyze query using Groq with structured output"""
        
        prompt_template = """
        You are a trading data query analyzer. Extract the following from the user's query:
        
        QUERY: {question}
        
        Extract:
        1. MAIN_INTENT: What is the primary intent of the query?
        2. REQUIRED_METRICS: List all trading metrics/indicators mentioned
        3. TIMEFRAME: What timeframe is mentioned?
        4. DATA_POINTS: Specific data points needed
        5. SYMBOL_INFERRED: Can you infer any specific trading symbol?
        6. EXPIRY_CONTEXT: Is there any expiry date context?
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question"],
            partial_variables={
                "format_instructions": self.query_parser.get_format_instructions()
            }
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            result = await chain.arun(question=question)
            parsed = self.query_parser.parse(result)
            return parsed
            
        except Exception as e:
            logger.warning(f"Failed to parse query analysis: {e}")
            # Fallback to direct API call
            return await self._analyze_query_direct(question)
    
    async def _analyze_query_direct(self, question: str) -> QueryAnalysis:
        """Fallback query analysis using direct API call"""
        
        prompt = f"""
        Analyze this trading query and return JSON:
        
        Query: "{question}"
        
        Return JSON with: main_intent, required_metrics (list), timeframe, 
        data_points (list), symbol_inferred (or null), expiry_context (or null)
        """
        
        completion = self.client.chat.completions.create(
            model=self.model_map['quick'],
            messages=[
                {"role": "system", "content": "You are a trading analyst. Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        
        # Convert to QueryAnalysis object
        return QueryAnalysis(**result)
    
    async def _fetch_relevant_data(self, query_analysis: QueryAnalysis, 
                                  symbol: str = None) -> Dict:
        """Fetch relevant data based on query analysis"""
        
        data = {}
        
        # Determine symbol
        target_symbol = symbol or query_analysis.symbol_inferred
        if not target_symbol:
            target_symbol = Config.SYMBOLS[0]
        
        metrics = [m.lower() for m in query_analysis.required_metrics]
        
        # Fetch data based on required metrics
        if any(m in ['volume', 'vol'] for m in metrics):
            spikes = await self.supabase.detect_volume_spikes(target_symbol)
            data['volume_spikes'] = spikes
        
        if any(m in ['gamma', 'greek', 'greeks'] for m in metrics):
            # Get latest gamma data
            response = self.supabase.client.table('gamma_data').select('*').eq(
                'symbol', target_symbol
            ).order('timestamp', desc=True).limit(1).execute()
            data['gamma_data'] = response.data[0] if response.data else {}
        
        if any(m in ['oi', 'open interest', 'openinterest'] for m in metrics):
            response = self.supabase.client.table('option_chain_data').select(
                'strike_price, option_type, oi, change_in_oi, iv, timestamp'
            ).eq('symbol', target_symbol).order('timestamp', desc=True).limit(100).execute()
            data['oi_data'] = response.data
        
        if any(m in ['price', 'ltp', 'last price'] for m in metrics):
            response = self.supabase.client.table('option_chain_data').select(
                'strike_price, option_type, ltp, timestamp'
            ).eq('symbol', target_symbol).order('timestamp', desc=True).limit(100).execute()
            data['price_data'] = response.data
        
        # Always fetch recent signals and predictions
        response = self.supabase.client.table('telegram_signals').select('*').eq(
            'symbol', target_symbol
        ).order('created_at', desc=True).limit(5).execute()
        data['signals'] = response.data
        
        response = self.supabase.client.table('ml_predictions').select('*').eq(
            'symbol', target_symbol
        ).order('trigger_time', desc=True).limit(5).execute()
        data['predictions'] = response.data
        
        return data
    
    async def _generate_response_with_groq(self, question: str, 
                                          data_context: Dict) -> str:
        """Generate response using Groq"""
        
        data_summary = self._create_data_summary(data_context)
        
        prompt_template = """
        You are EXPERT OPTIONS ANALYST with 20+ years of experience.
        
        USER QUESTION: {question}
        
        AVAILABLE DATA:
        {data_summary}
        
        Provide a PROFESSIONAL TRADING ANALYSIS with these sections:
        
        1. 📊 DATA INSIGHTS:
           - Key observations from the data
           - Patterns, anomalies, or unusual activities
           - Statistical significance of findings
        
        2. 🎯 TRADING IMPLICATIONS:
           - What this means for traders
           - Potential opportunities
           - Risk factors to consider
        
        3. 📈 ACTIONABLE RECOMMENDATIONS:
           - Specific trading actions (if any)
           - Position sizing suggestions
           - Entry/exit guidelines
        
        4. ⚠️ RISK ASSESSMENT:
           - Key risks identified
           - Risk mitigation strategies
           - Worst-case scenarios
        
        5. 🔮 FORWARD OUTLOOK:
           - What to watch for next
           - Key levels to monitor
           - Timeframe for the analysis
        
        Format with clear section headings and bullet points.
        Be specific, data-driven, and professional.
        """
        
        prompt = PromptTemplate(
            input_variables=["question", "data_summary"],
            template=prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            response = await chain.arun(
                question=question,
                data_summary=data_summary
            )
            
            # Track tokens (approximate)
            self.usage_stats['total_tokens'] += len(response.split()) * 1.3
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _create_data_summary(self, data_context: Dict) -> str:
        """Create comprehensive data summary"""
        
        summary_parts = []
        
        for key, data in data_context.items():
            if not data:
                continue
                
            if isinstance(data, list):
                if key == 'volume_spikes' and data:
                    summary_parts.append(f"📊 VOLUME SPIKES ({len(data)}):")
                    for spike in data[:3]:
                        summary_parts.append(
                            f"  • {spike.get('strike_price')} {spike.get('option_type')}: "
                            f"{spike.get('current_volume'):,} (avg: {spike.get('avg_volume'):,}, "
                            f"ratio: {spike.get('volume_ratio'):.2f}x)"
                        )
                
                elif key in ['oi_data', 'price_data'] and data:
                    df = pd.DataFrame(data)
                    summary_parts.append(f"📈 {key.upper()} ({len(df)} records):")
                    if 'option_type' in df.columns:
                        calls = df[df['option_type'] == 'CE']
                        puts = df[df['option_type'] == 'PE']
                        if len(calls) > 0:
                            summary_parts.append(f"  • Calls: {len(calls)} strikes, avg OI: {calls['oi'].mean():,.0f}")
                        if len(puts) > 0:
                            summary_parts.append(f"  • Puts: {len(puts)} strikes, avg OI: {puts['oi'].mean():,.0f}")
                
                elif key in ['signals', 'predictions'] and data:
                    summary_parts.append(f"🎯 {key.upper()} ({len(data)}):")
                    for item in data[:2]:
                        if key == 'signals':
                            summary_parts.append(
                                f"  • {item.get('signal_type')} at {item.get('created_at')}"
                            )
                        else:
                            summary_parts.append(
                                f"  • {item.get('prediction_type')}: {item.get('prediction_value'):.4f}"
                            )
            
            elif isinstance(data, dict):
                if key == 'gamma_data' and data:
                    summary_parts.append("🧮 GAMMA DATA:")
                    summary_parts.append(f"  • Gamma Exposure: {data.get('gamma_exposure', 0):,.2f}")
                    summary_parts.append(f"  • Total Gamma: {data.get('total_gamma', 0):,.2f}")
                    summary_parts.append(f"  • Call/Put Gamma: {data.get('call_gamma', 0):,.2f}/{data.get('put_gamma', 0):,.2f}")
        
        if not summary_parts:
            return "No data available for analysis."
        
        return "\n".join(summary_parts)
    
    async def generate_trading_signal_analysis(self, spike_detection: Dict, 
                                              news_context: List[Dict]) -> TradingSignal:
        """Generate trading signal analysis using Groq"""
        
        self.usage_stats['total_requests'] += 1
        
        # Format input data
        spike_summary = self._format_spike_for_analysis(spike_detection)
        news_summary = self._format_news_for_analysis(news_context)
        
        prompt_template = """
        You are a SENIOR OPTIONS TRADING STRATEGIST at a hedge fund.
        
        SPIKE DETECTION ALERT:
        {spike_summary}
        
        RELEVANT MARKET NEWS:
        {news_summary}
        
        Based on this spike detection and market context, provide a COMPREHENSIVE TRADING SIGNAL ANALYSIS.
        
        {format_instructions}
        
        Guidelines:
        1. Signal should be based on data, not guesswork
        2. Confidence should reflect data quality and spike strength
        3. Price targets should be realistic based on technical levels
        4. Stop loss should protect capital (1-2% typically)
        5. Be conservative with position sizing (1-3% of portfolio)
        6. Consider time to expiry in your timeframe
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["spike_summary", "news_summary"],
            partial_variables={
                "format_instructions": self.signal_parser.get_format_instructions()
            }
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            result = await chain.arun(
                spike_summary=spike_summary,
                news_summary=news_summary
            )
            
            signal = self.signal_parser.parse(result)
            
            # Track usage
            self.usage_stats['total_tokens'] += len(json.dumps(signal.dict())) * 1.3
            self.usage_stats['models_used'][self.model_name] = \
                self.usage_stats['models_used'].get(self.model_name, 0) + 1
            
            logger.info(f"Generated trading signal: {signal.signal_type} "
                       f"(confidence: {signal.confidence:.2%})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal analysis: {e}")
            # Return conservative hold signal
            return TradingSignal(
                signal_type="HOLD",
                confidence=0.3,
                rationale=f"Error in AI analysis: {str(e)}. Defaulting to hold.",
                key_risk_factors=["AI system error", "Uncertain market conditions"],
                price_target=0,
                stop_loss=0,
                timeframe_hours=24,
                position_size_percent=0,
                entry_strategy="Wait for system recovery",
                exit_strategy="Manual monitoring required",
                monitoring_parameters=["System status", "Data quality"],
                alternate_scenarios=["System recovery", "Manual analysis needed"]
            )
    
    def _format_spike_for_analysis(self, spike_detection: Dict) -> str:
        """Format spike detection data for analysis"""
        
        lines = [
            f"Symbol: {spike_detection.get('symbol', 'N/A')}",
            f"Expiry Date: {spike_detection.get('expiry_date', 'N/A')}",
            f"Detection Confidence: {spike_detection.get('confidence', 0):.2%}",
            f"Underlying Price: {spike_detection.get('underlying_price', 0):.2f}",
            f"Total Volume: {spike_detection.get('total_volume', 0):,}",
            f"Total OI: {spike_detection.get('total_oi', 0):,}",
            f"Gamma Exposure: {spike_detection.get('gamma_exposure', 0):.2f}",
            f"Call Gamma: {spike_detection.get('call_gamma', 0):.2f}",
            f"Put Gamma: {spike_detection.get('put_gamma', 0):.2f}",
            f"Volume Spikes Detected: {spike_detection.get('volume_spikes_count', 0)}",
            f"Days to Expiry: {spike_detection.get('days_to_expiry', 0)}",
            f"Spike Score: {spike_detection.get('spike_score', 0):.2f}",
            f"Detection Time: {spike_detection.get('detection_time', 'N/A')}",
            f"AI Model Used: {spike_detection.get('ai_model', 'Groq AI')}"
        ]
        
        return "\n".join(lines)
    
    def _format_news_for_analysis(self, news_context: List[Dict]) -> str:
        """Format news context for analysis"""
        
        if not news_context:
            return "No relevant news available."
        
        lines = ["Relevant Market News:"]
        
        for i, news in enumerate(news_context[:3], 1):
            title = news.get('title', 'No title')[:80]
            sentiment = news.get('sentiment', {}).get('category', 'neutral').upper()
            score = news.get('sentiment', {}).get('score', 0)
            relevance = news.get('relevance_score', 0)
            
            lines.append(f"{i}. {title}")
            lines.append(f"   Sentiment: {sentiment} (Score: {score:.3f}, Relevance: {relevance:.2f})")
            
            if 'keywords_found' in news and news['keywords_found']:
                lines.append(f"   Keywords: {', '.join(news['keywords_found'][:5])}")
        
        return "\n".join(lines)
    
    async def get_market_analysis(self, symbols: List[str] = None) -> Dict:
        """Get comprehensive market analysis for multiple symbols"""
        
        if symbols is None:
            symbols = Config.get_trading_symbols()
        
        analyses = {}
        
        for symbol in symbols:
            # Fetch latest data
            data = await self._get_symbol_summary(symbol)
            analysis = await self._analyze_symbol(symbol, data)
            analyses[symbol] = analysis.dict() if hasattr(analysis, 'dict') else analysis
        
        # Generate overall market view
        overall = await self._generate_overall_market_view(analyses)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol_analyses': analyses,
            'overall_market_view': overall,
            'model_used': self.model_name
        }
    
    async def _get_symbol_summary(self, symbol: str) -> Dict:
        """Get summary data for a symbol"""
        
        data = {}
        
        # Get latest gamma data
        gamma_resp = self.supabase.client.table('gamma_data').select('*').eq(
            'symbol', symbol
        ).order('timestamp', desc=True).limit(1).execute()
        
        data['gamma'] = gamma_resp.data[0] if gamma_resp.data else {}
        
        # Get recent spikes
        spikes_resp = self.supabase.client.table('volume_spikes').select('*').eq(
            'symbol', symbol
        ).order('timestamp', desc=True).limit(5).execute()
        
        data['spikes'] = spikes_resp.data
        
        # Get latest predictions
        pred_resp = self.supabase.client.table('ml_predictions').select('*').eq(
            'symbol', symbol
        ).order('trigger_time', desc=True).limit(3).execute()
        
        data['predictions'] = pred_resp.data
        
        return data
    
    async def _analyze_symbol(self, symbol: str, data: Dict) -> MarketAnalysis:
        """Analyze a single symbol"""
        
        data_str = json.dumps(data, indent=2)
        
        prompt_template = """
        Analyze this trading data for {symbol}:
        
        {data_str}
        
        Provide a concise market analysis.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["symbol", "data_str"],
            partial_variables={
                "format_instructions": self.analysis_parser.get_format_instructions()
            }
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            result = await chain.arun(symbol=symbol, data_str=data_str)
            return self.analysis_parser.parse(result)
            
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            # Return default analysis
            return MarketAnalysis(
                condition="Unknown",
                bias="neutral",
                risk_level="medium",
                key_levels={},
                summary=f"Error in analysis: {str(e)}"
            )
    
    async def _generate_overall_market_view(self, symbol_analyses: Dict) -> str:
        """Generate overall market view"""
        
        analyses_summary = []
        for symbol, analysis in symbol_analyses.items():
            if isinstance(analysis, dict):
                analyses_summary.append(
                    f"{symbol}: {analysis.get('condition', 'Unknown')} "
                    f"({analysis.get('bias', 'neutral')}, "
                    f"risk: {analysis.get('risk_level', 'medium')})"
                )
        
        prompt = f"""
        Based on these individual symbol analyses, provide an overall market view:
        
        {chr(10).join(analyses_summary)}
        
        Consider:
        1. Overall market sentiment
        2. Key themes across symbols
        3. Market-wide risks
        4. Trading opportunities
        5. Recommendations for the day
        
        Provide a concise, professional overview.
        """
        
        completion = self.client.chat.completions.create(
            model=self.model_map['analysis'],
            messages=[
                {"role": "system", "content": "You are a market strategist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return self.usage_stats.copy()
