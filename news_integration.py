import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from newspaper import Article, ArticleException
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from config import Config

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """News gathering and sentiment analysis"""
    
    def __init__(self, newsapi_key: str = None):
        self.newsapi_key = newsapi_key or Config.NEWSAPI_KEY
        self.sia = SentimentIntensityAnalyzer()
        
        # Download NLTK data if not present
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        # Keywords for options trading relevance
        self.options_keywords = [
            'expiry', 'options', 'volatility', 'gamma', 'delta', 'theta', 'vega',
            'hedging', 'rollover', 'unwinding', 'straddle', 'strangle', 'spread',
            'premium', 'strike', 'implied volatility', 'iv', 'greeks',
            'derivatives', 'f&o', 'futures and options', 'max pain',
            'put call ratio', 'pcr', 'open interest', 'oi'
        ]
        
        self.market_keywords = [
            'nifty', 'bank nifty', 'sensex', 'stock market', 'indian market',
            'rbi', 'sebi', 'fii', 'dii', 'foreign institutional investors',
            'domestic institutional investors', 'market sentiment',
            'economic data', 'gdp', 'inflation', 'interest rates', 'monetary policy'
        ]
    
    async def fetch_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Fetch news for a symbol from multiple sources"""
        news_items = []
        
        # Fetch from NewsAPI if key is available
        if self.newsapi_key:
            news_items.extend(await self._fetch_newsapi(symbol, hours_back))
        
        # Additional sources can be added here
        # news_items.extend(await self._fetch_alternative_source(symbol, hours_back))
        
        # Deduplicate news items
        deduplicated = self._deduplicate_news(news_items)
        
        return deduplicated[:Config.MAX_NEWS_ITEMS]
    
    async def _fetch_newsapi(self, symbol: str, hours_back: int) -> List[Dict]:
        """Fetch news from NewsAPI"""
        news_items = []
        
        from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%dT%H:%M:%S')
        
        url = 'https://newsapi.org/v2/everything'
        
        # Search queries
        queries = [
            f"{symbol} stock",
            f"{symbol} options",
            f"{symbol} derivatives",
            f"NSE {symbol}",
            f"{symbol} expiry"
        ]
        
        async with aiohttp.ClientSession() as session:
            for query in queries:
                params = {
                    'q': query,
                    'from': from_date,
                    'sortBy': 'relevancy',
                    'language': 'en',
                    'apiKey': self.newsapi_key,
                    'pageSize': 10
                }
                
                try:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for article in data.get('articles', []):
                                # Extract content if URL is available
                                content = ""
                                if article.get('url'):
                                    content = await self._extract_article_content(article['url'])
                                
                                news_item = {
                                    'symbol': symbol,
                                    'title': article.get('title', ''),
                                    'description': article.get('description', ''),
                                    'url': article.get('url', ''),
                                    'source': article.get('source', {}).get('name', ''),
                                    'published_at': article.get('publishedAt', ''),
                                    'content': content[:2000] if content else '',  # Limit content size
                                    'raw_data': article
                                }
                                news_items.append(news_item)
                                
                        else:
                            logger.warning(f"NewsAPI returned status {response.status}")
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout fetching news for query: {query}")
                except Exception as e:
                    logger.error(f"Error fetching from NewsAPI: {e}")
        
        return news_items
    
    async def _extract_article_content(self, url: str) -> str:
        """Extract full article content from URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Clean and limit content
            content = article.text.strip()
            if len(content) > 5000:
                content = content[:5000] + "..."
                
            return content
            
        except ArticleException as e:
            logger.debug(f"Could not parse article {url}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting article content: {e}")
            return ""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using VADER"""
        if not text or len(text.strip()) < 10:
            return {
                'score': 0.0,
                'category': 'neutral',
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            }
        
        sentiment = self.sia.polarity_scores(text)
        
        # Categorize sentiment
        compound = sentiment['compound']
        if compound >= 0.05:
            category = 'positive'
        elif compound <= -0.05:
            category = 'negative'
        else:
            category = 'neutral'
        
        return {
            'score': compound,
            'category': category,
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'compound': compound
        }
    
    async def get_relevant_news(self, symbol: str, 
                               additional_keywords: List[str] = None) -> List[Dict]:
        """Get news relevant to options trading and expiry spikes"""
        
        # Combine keywords
        all_keywords = self.options_keywords + self.market_keywords
        if additional_keywords:
            all_keywords.extend(additional_keywords)
        
        # Fetch news
        all_news = await self.fetch_news(symbol)
        relevant_news = []
        
        for news in all_news:
            # Combine text for analysis
            combined_text = f"{news['title']} {news['description']} {news.get('content', '')}"
            combined_text_lower = combined_text.lower()
            
            # Check for keyword matches
            keyword_matches = []
            for keyword in all_keywords:
                if keyword.lower() in combined_text_lower:
                    keyword_matches.append(keyword)
            
            # Calculate relevance score
            if keyword_matches:
                sentiment = self.analyze_sentiment(combined_text)
                
                # Calculate relevance score based on:
                # 1. Number of keyword matches
                # 2. Recency (more recent = higher score)
                # 3. Source credibility
                # 4. Sentiment strength
                
                keyword_score = len(keyword_matches) / len(all_keywords)
                
                # Recency score
                published_at = news.get('published_at')
                recency_score = 1.0
                if published_at:
                    try:
                        pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        hours_ago = (datetime.now(pub_time.tzinfo) - pub_time).total_seconds() / 3600
                        recency_score = max(0, 1 - (hours_ago / 48))  # Decay over 48 hours
                    except:
                        pass
                
                # Source credibility (simple mapping)
                source = news.get('source', '').lower()
                source_score = 0.5  # Default
                credible_sources = ['reuters', 'bloomberg', 'economic times', 'moneycontrol']
                if any(cred in source for cred in credible_sources):
                    source_score = 1.0
                
                # Sentiment strength
                sentiment_strength = abs(sentiment['score'])
                
                # Combined relevance score
                relevance_score = (
                    keyword_score * 0.4 +
                    recency_score * 0.3 +
                    source_score * 0.2 +
                    sentiment_strength * 0.1
                )
                
                news['sentiment'] = sentiment
                news['keywords_found'] = keyword_matches
                news['relevance_score'] = relevance_score
                news['combined_text'] = combined_text[:500]  # Store for reference
                
                relevant_news.append(news)
        
        # Sort by relevance score
        relevant_news.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Found {len(relevant_news)} relevant news items for {symbol}")
        
        return relevant_news
    
    def _deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on title similarity"""
        if not news_items:
            return []
        
        # Simple deduplication based on title
        seen_titles = set()
        deduplicated = []
        
        for news in news_items:
            title = news.get('title', '').strip().lower()
            
            # Check if similar title already seen
            is_duplicate = False
            for seen_title in seen_titles:
                # Simple similarity check (can be improved)
                if title and seen_title and (
                    title in seen_title or 
                    seen_title in title or
                    self._jaccard_similarity(title, seen_title) > 0.8
                ):
                    is_duplicate = True
                    break
            
            if not is_duplicate and title:
                seen_titles.add(title)
                deduplicated.append(news)
        
        return deduplicated
    
    def _jaccard_similarity(self, str1: str, str2: str) -> float:
        """Calculate Jaccard similarity between two strings"""
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    async def get_market_sentiment(self, symbols: List[str] = None) -> Dict:
        """Get overall market sentiment from news"""
        if symbols is None:
            symbols = Config.get_trading_symbols()
        
        all_news = []
        
        # Fetch news for all symbols
        tasks = [self.get_relevant_news(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching news for {symbol}: {result}")
            elif result:
                all_news.extend(result)
        
        if not all_news:
            return {
                'overall_sentiment': 'neutral',
                'score': 0.0,
                'news_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Aggregate sentiment
        sentiment_scores = [n['sentiment']['score'] for n in all_news]
        avg_score = np.mean(sentiment_scores)
        
        # Categorize
        if avg_score >= 0.05:
            overall_sentiment = 'positive'
        elif avg_score <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Count by category
        categories = [n['sentiment']['category'] for n in all_news]
        positive_count = categories.count('positive')
        negative_count = categories.count('negative')
        neutral_count = categories.count('neutral')
        
        return {
            'overall_sentiment': overall_sentiment,
            'score': float(avg_score),
            'news_count': len(all_news),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'top_stories': all_news[:3]  # Top 3 most relevant
        }
