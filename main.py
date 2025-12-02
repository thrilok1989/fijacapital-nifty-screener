#!/usr/bin/env python3
"""
AI Trading System - Expiry Spike Detector
Main orchestrator using Groq AI
"""

import asyncio
import schedule
import time
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import traceback
from colorlog import ColoredFormatter

from config import Config
from dhan_client import DhanClient
from supabase_manager import SupabaseManager
from ml_detector import ExpirySpikeMLDetector
from news_integration import NewsAnalyzer
from groq_ai_system import GroqAIAnalyzer
from telegram_notifier import TelegramNotifier
from groq_monitor import GroqUsageMonitor

# Setup logging
def setup_logging():
    """Setup colored logging"""
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set specific log levels
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)

class TradingSystemOrchestrator:
    """Main orchestrator for the trading system"""
    
    def __init__(self):
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        errors = Config.validate_config()
        if errors:
            self.logger.error("Configuration errors:")
            for error in errors:
                self.logger.error(f"  • {error}")
            raise ValueError("Invalid configuration. Please check your .env file.")
        
        # Initialize components
        self.supabase = SupabaseManager()
        self.dhan_client = None
        self.ml_detector = ExpirySpikeMLDetector()
        self.news_analyzer = NewsAnalyzer()
        self.ai_system = GroqAIAnalyzer(supabase_manager=self.supabase)
        self.telegram = TelegramNotifier()
        self.usage_monitor = GroqUsageMonitor()
        
        # System state
        self.is_running = False
        self.analysis_count = 0
        self.signal_count = 0
        self.last_cycle_time = None
        self.start_time = datetime.now()
        
        # Performance tracking
        self.performance_stats = {
            'cycles_completed': 0,
            'spikes_detected': 0,
            'signals_generated': 0,
            'errors': 0,
            'total_processing_time': 0.0
        }
        
        self.logger.info("Trading System Orchestrator initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing trading system...")
            
            # Initialize Dhan client
            self.dhan_client = DhanClient()
            
            # Test database connection
            await self.supabase.log_system_event(
                'INFO', 'System', 'Initialization started'
            )
            
            # Test AI system
            test_query = "Test connection"
            await self.ai_system.query_data(test_query)
            
            # Send startup notification
            await self.telegram.send_alert(
                "SYSTEM STARTUP",
                f"AI Trading System initialized successfully\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Symbols: {', '.join(Config.SYMBOLS)}\n"
                f"AI Model: {Config.GROQ_MODEL}",
                notification=False
            )
            
            self.logger.info("✅ System initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            await self.telegram.send_alert(
                "SYSTEM ERROR",
                f"Initialization failed: {str(e)[:200]}"
            )
            return False
    
    async def run_detection_cycle(self):
        """Run one complete detection cycle"""
        if not Config.is_market_open():
            self.logger.debug("Market is closed, skipping cycle")
            return
        
        cycle_start = time.time()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🚀 Starting detection cycle #{self.analysis_count + 1}")
        self.logger.info(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}")
        
        try:
            for symbol in Config.SYMBOLS:
                await self._process_symbol(symbol)
            
            # Update statistics
            self.analysis_count += 1
            self.last_cycle_time = datetime.now()
            self.performance_stats['cycles_completed'] += 1
            self.performance_stats['total_processing_time'] += time.time() - cycle_start
            
            self.logger.info(f"✅ Cycle completed in {time.time() - cycle_start:.2f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error in detection cycle: {e}")
            self.performance_stats['errors'] += 1
            await self.supabase.log_system_event(
                'ERROR', 'Orchestrator', f'Detection cycle failed: {str(e)}',
                {'traceback': traceback.format_exc()}
            )
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol"""
        self.logger.info(f"\n🔍 Processing {symbol}...")
        
        try:
            # Get nearest expiry
            expiry_date = self.dhan_client.get_nearest_expiry()
            
            # Fetch option chain data
            async with self.dhan_client as client:
                option_data = await client.get_option_chain(symbol, expiry_date)
            
            if not option_data:
                self.logger.warning(f"⚠️ No option data for {symbol}")
                return
            
            # Store in database
            await self.supabase.store_option_chain(option_data)
            
            # Calculate gamma sequence
            gamma_data = await self.supabase.calculate_gamma_sequence(symbol, expiry_date)
            
            # Detect volume spikes
            volume_spikes = await self.supabase.detect_volume_spikes(symbol)
            
            # Fetch historical data
            async with self.dhan_client as client:
                historical_data = await client.get_historical_data(
                    symbol, interval='5minute',
                    from_date=(datetime.now() - timedelta(days=Config.FEATURE_WINDOW_DAYS)).strftime('%Y-%m-%d')
                )
            
            # Extract features and predict
            if len(historical_data) >= 20:
                features = self.ml_detector.extract_features(
                    historical_data, option_data, gamma_data
                )
                
                is_spike, confidence = self.ml_detector.predict_spike(features)
                
                self.logger.info(f"📊 ML Prediction: Spike={is_spike}, Confidence={confidence:.2%}")
                
                # Store prediction
                prediction_data = {
                    'symbol': symbol,
                    'expiry_date': expiry_date,
                    'prediction_type': 'expiry_spike',
                    'prediction_value': confidence,
                    'confidence': confidence,
                    'features': features.tolist() if hasattr(features, 'tolist') else features,
                    'model_version': 'xgboost_1.0'
                }
                await self.supabase.store_ml_prediction(prediction_data)
                
                # Check for spike
                if is_spike and confidence > Config.SPIKE_THRESHOLD:
                    self.performance_stats['spikes_detected'] += 1
                    await self._handle_spike_detection(
                        symbol, expiry_date, option_data, gamma_data,
                        volume_spikes, confidence, features
                    )
                elif confidence > Config.ANALYSIS_CONFIDENCE_THRESHOLD:
                    # Perform light analysis for moderate activity
                    await self._perform_light_analysis(
                        symbol, expiry_date, option_data, gamma_data, confidence
                    )
            else:
                self.logger.warning(f"⚠️ Insufficient historical data for {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing {symbol}: {e}")
            await self.supabase.log_system_event(
                'ERROR', 'Orchestrator', f'Error processing {symbol}: {str(e)}',
                {'symbol': symbol, 'traceback': traceback.format_exc()}
            )
    
    async def _handle_spike_detection(self, symbol: str, expiry_date: str,
                                     option_data: Dict, gamma_data: Dict,
                                     volume_spikes: List[Dict], 
                                     confidence: float, features: Any):
        """Handle spike detection with AI analysis"""
        
        self.logger.info(f"🚨 SPIKE DETECTED for {symbol} (Confidence: {confidence:.2%})")
        
        try:
            # Get relevant news
            relevant_news = await self.news_analyzer.get_relevant_news(symbol)
            
            # Store news in database
            if relevant_news:
                await self.supabase.store_news(relevant_news)
            
            # Prepare spike data
            spike_data = {
                'symbol': symbol,
                'expiry_date': expiry_date,
                'confidence': confidence,
                'underlying_price': option_data.get('underlying_price', 0),
                'total_volume': option_data.get('total_volume', 0),
                'total_oi': option_data.get('total_oi', 0),
                'gamma_exposure': gamma_data.get('gamma_exposure', 0),
                'call_gamma': gamma_data.get('call_gamma', 0),
                'put_gamma': gamma_data.get('put_gamma', 0),
                'put_call_ratio_volume': option_data.get('put_call_ratio_volume', 0),
                'put_call_ratio_oi': option_data.get('put_call_ratio_oi', 0),
                'spike_score': confidence,
                'volume_spikes_count': len(volume_spikes),
                'days_to_expiry': (datetime.strptime(expiry_date, '%Y-%m-%d') - datetime.now()).days,
                'detection_time': datetime.now().isoformat(),
                'ai_model': Config.GROQ_MODEL
            }
            
            # Generate AI signal analysis
            analysis_start = time.time()
            signal_analysis = await self.ai_system.generate_trading_signal_analysis(
                spike_data, relevant_news
            )
            analysis_time = time.time() - analysis_start
            
            # Track usage
            self.usage_monitor.record_usage(
                Config.GROQ_MODEL,
                tokens=len(json.dumps(signal_analysis.dict())) * 1.3,  # Approximate
                duration=analysis_time,
                success=True,
                details={'symbol': symbol, 'type': 'signal_analysis'}
            )
            
            # Prepare final signal
            signal = {
                **spike_data,
                **signal_analysis.dict(),
                'news_context': relevant_news[:3],
                'volume_spikes': volume_spikes[:5],
                'analysis_time_seconds': analysis_time,
                'model_version': Config.GROQ_MODEL
            }
            
            # Store signal
            signal_id = await self.supabase.store_signal(signal)
            if signal_id:
                self.performance_stats['signals_generated'] += 1
                self.signal_count += 1
            
            # Send Telegram notification
            await self.telegram.send_signal(signal)
            
            self.logger.info(f"✅ Signal generated: {signal_analysis.signal_type} "
                           f"(Confidence: {signal_analysis.confidence:.2%})")
            
        except Exception as e:
            self.logger.error(f"❌ Error in spike handling: {e}")
            await self.supabase.log_system_event(
                'ERROR', 'Orchestrator', f'Spike handling failed for {symbol}: {str(e)}',
                {'symbol': symbol, 'traceback': traceback.format_exc()}
            )
    
    async def _perform_light_analysis(self, symbol: str, expiry_date: str,
                                     option_data: Dict, gamma_data: Dict,
                                     confidence: float):
        """Perform light analysis for moderate activity"""
        self.logger.debug(f"📈 Moderate activity for {symbol} (Confidence: {confidence:.2%})")
        
        # Store for monitoring
        await self.supabase.store_ml_prediction({
            'symbol': symbol,
            'expiry_date': expiry_date,
            'prediction_type': 'moderate_activity',
            'prediction_value': confidence,
            'confidence': confidence,
            'model_version': 'monitoring'
        })
    
    async def ai_query_handler(self, question: str, symbol: str = None):
        """Handle AI queries from user"""
        try:
            if not symbol:
                symbol = Config.SYMBOLS[0]
            
            self.logger.info(f"🤖 Processing AI query: {question}")
            
            query_start = time.time()
            result = await self.ai_system.query_data(question, symbol)
            query_time = time.time() - query_start
            
            # Track usage
            self.usage_monitor.record_usage(
                Config.GROQ_MODEL,
                tokens=len(json.dumps(result)) * 1.3,
                duration=query_time,
                success=True,
                details={'type': 'user_query', 'question': question}
            )
            
            # Send to Telegram
            response = result.get('response', 'No response generated')
            await self.telegram.send_alert(
                "AI QUERY RESULT",
                f"Q: {question}\n\nA: {response[:3000]}..."
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ AI query failed: {e}")
            await self.telegram.send_alert(
                "AI QUERY ERROR",
                f"Query failed: {str(e)[:200]}"
            )
            return {'error': str(e)}
    
    async def get_system_status(self) -> Dict:
        """Get system status"""
        uptime = datetime.now() - self.start_time
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime_seconds': uptime.total_seconds(),
            'analysis_count': self.analysis_count,
            'signal_count': self.signal_count,
            'last_cycle': self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            'market_open': Config.is_market_open(),
            'symbols_monitored': Config.SYMBOLS,
            'performance_stats': self.performance_stats,
            'groq_usage': self.usage_monitor.get_summary(),
            'config': {
                'check_interval': Config.CHECK_INTERVAL,
                'spike_threshold': Config.SPIKE_THRESHOLD,
                'groq_model': Config.GROQ_MODEL
            }
        }
    
    def start_scheduler(self):
        """Start scheduled detection cycles"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🚀 Starting AI Trading System Scheduler")
        self.logger.info(f"📊 Symbols: {', '.join(Config.SYMBOLS)}")
        self.logger.info(f"⏱️ Check Interval: {Config.CHECK_INTERVAL} seconds")
        self.logger.info(f"🎯 Spike Threshold: {Config.SPIKE_THRESHOLD}")
        self.logger.info(f"🤖 AI Model: {Config.GROQ_MODEL}")
        self.logger.info(f"⏰ Market Hours: {Config.MARKET_OPEN_TIME} - {Config.MARKET_CLOSE_TIME}")
        self.logger.info(f"{'='*60}")
        
        self.is_running = True
        
        # Schedule detection cycles
        schedule.every(Config.CHECK_INTERVAL).seconds.do(
            lambda: asyncio.create_task(self.run_detection_cycle())
        )
        
        # Schedule hourly market overview
        schedule.every(1).hours.do(
            lambda: asyncio.create_task(self._send_market_overview())
        )
        
        # Schedule daily cleanup (3:45 PM)
        schedule.every().day.at("15:45").do(
            lambda: asyncio.create_task(self._cleanup_old_data())
        )
        
        # Schedule daily report (3:30 PM)
        schedule.every().day.at("15:30").do(
            lambda: asyncio.create_task(self._send_daily_report())
        )
        
        # Schedule usage report (every 6 hours)
        schedule.every(6).hours.do(
            lambda: asyncio.create_task(self._send_usage_report())
        )
        
        self.logger.info("✅ Scheduler started successfully")
        
        # Run scheduler loop
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
    
    async def _send_market_overview(self):
        """Send market overview"""
        try:
            overview = await self.ai_system.get_market_analysis()
            await self.telegram.send_alert(
                "MARKET OVERVIEW",
                f"Market analysis generated at {datetime.now().strftime('%H:%M')}"
            )
        except Exception as e:
            self.logger.error(f"Error sending market overview: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old data"""
        try:
            await self.supabase.cleanup_old_data(days_to_keep=30)
            self.logger.info("Old data cleanup completed")
        except Exception as e:
            self.logger.error(f"Error in data cleanup: {e}")
    
    async def _send_daily_report(self):
        """Send daily trading report"""
        try:
            status = await self.get_system_status()
            
            report = f"""
📊 DAILY TRADING REPORT

📅 Date: {datetime.now().strftime('%Y-%m-%d')}
⏰ Uptime: {status['uptime_seconds']/3600:.1f} hours
🔍 Analyses: {status['analysis_count']}
🚨 Spikes: {status['performance_stats']['spikes_detected']}
📈 Signals: {status['performance_stats']['signals_generated']}
⚠️ Errors: {status['performance_stats']['errors']}

🤖 AI Usage:
• Requests: {status['groq_usage']['total_requests']}
• Tokens: {status['groq_usage']['total_tokens']:,}
• Cost: ${status['groq_usage']['total_cost_usd']:.4f}
• Error Rate: {status['groq_usage']['error_rate']:.2%}

💡 System running normally.
            """
            
            await self.telegram.send_alert("DAILY REPORT", report)
            
        except Exception as e:
            self.logger.error(f"Error sending daily report: {e}")
    
    async def _send_usage_report(self):
        """Send usage report"""
        try:
            usage = self.usage_monitor.get_summary()
            
            report = f"""
🤖 GROQ USAGE REPORT

📊 Total Requests: {usage['total_requests']:,}
🔤 Total Tokens: {usage['total_tokens']:,}
💰 Estimated Cost: ${usage['total_cost_usd']:.4f}
⏱️ Total Time: {usage['total_time_seconds']:.0f}s
⚠️ Errors: {usage['total_errors']}

📈 Averages:
• Time/Request: {usage['avg_time_per_request']:.2f}s
• Tokens/Request: {usage['avg_tokens_per_request']:.0f}
• Error Rate: {usage['error_rate']:.2%}

🔧 Models Used: {', '.join(usage['models_used'])}
            """
            
            await self.telegram.send_alert("USAGE REPORT", report, notification=False)
            
        except Exception as e:
            self.logger.error(f"Error sending usage report: {e}")
    
    def stop(self):
        """Stop the system"""
        self.is_running = False
        schedule.clear()
        self.logger.info("System stopped")

async def main():
    """Main entry point"""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "init":
            print("Initializing system...")
            orchestrator = TradingSystemOrchestrator()
            success = await orchestrator.initialize()
            if success:
                print("✅ System initialized successfully")
            else:
                print("❌ System initialization failed")
            return
        
        elif command == "run":
            print("Running single detection cycle...")
            orchestrator = TradingSystemOrchestrator()
            await orchestrator.initialize()
            await orchestrator.run_detection_cycle()
            return
        
        elif command == "start":
            print("Starting trading system...")
            orchestrator = TradingSystemOrchestrator()
            await orchestrator.initialize()
            orchestrator.start_scheduler()
            return
        
        elif command == "query":
            if len(sys.argv) < 3:
                print("Usage: python main.py query 'your question' [symbol]")
                return
            
            question = sys.argv[2]
            symbol = sys.argv[3] if len(sys.argv) > 3 else None
            
            print(f"Processing query: {question}")
            orchestrator = TradingSystemOrchestrator()
            await orchestrator.initialize()
            result = await orchestrator.ai_query_handler(question, symbol)
            print(f"Result: {json.dumps(result, indent=2)}")
            return
        
        elif command == "status":
            orchestrator = TradingSystemOrchestrator()
            await orchestrator.initialize()
            status = await orchestrator.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        elif command == "usage":
            orchestrator = TradingSystemOrchestrator()
            orchestrator.usage_monitor.print_summary()
            return
        
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  init     - Initialize system")
            print("  run      - Run single detection cycle")
            print("  start    - Start scheduler")
            print("  query    - Ask AI a question")
            print("  status   - Show system status")
            print("  usage    - Show Groq usage stats")
            return
    
    # Interactive mode
    print("\n" + "="*60)
    print("🤖 AI TRADING SYSTEM - EXPIRY SPIKE DETECTOR")
    print("="*60)
    print("\nCommands: init, run, start, query, status, usage, exit")
    
    orchestrator = None
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'exit':
                if orchestrator:
                    orchestrator.stop()
                print("Goodbye!")
                break
            
            elif cmd == 'init':
                orchestrator = TradingSystemOrchestrator()
                success = await orchestrator.initialize()
                if success:
                    print("✅ System initialized")
                else:
                    print("❌ Initialization failed")
            
            elif cmd == 'run':
                if not orchestrator:
                    print("Please run 'init' first")
                else:
                    await orchestrator.run_detection_cycle()
            
            elif cmd == 'start':
                if not orchestrator:
                    orchestrator = TradingSystemOrchestrator()
                    await orchestrator.initialize()
                orchestrator.start_scheduler()
                break  # Start scheduler blocks
            
            elif cmd == 'query':
                if not orchestrator:
                    orchestrator = TradingSystemOrchestrator()
                    await orchestrator.initialize()
                
                question = input("Enter your question: ")
                symbol = input(f"Symbol ({Config.SYMBOLS[0]}): ").strip()
                if not symbol:
                    symbol = None
                
                result = await orchestrator.ai_query_handler(question, symbol)
                print(f"\nResult: {json.dumps(result, indent=2)}")
            
            elif cmd == 'status':
                if not orchestrator:
                    orchestrator = TradingSystemOrchestrator()
                    await orchestrator.initialize()
                
                status = await orchestrator.get_system_status()
                print(json.dumps(status, indent=2))
            
            elif cmd == 'usage':
                if not orchestrator:
                    orchestrator = TradingSystemOrchestrator()
                orchestrator.usage_monitor.print_summary()
            
            else:
                print("Unknown command. Available: init, run, start, query, status, usage, exit")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
            if orchestrator:
                orchestrator.stop()
            break
        
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
