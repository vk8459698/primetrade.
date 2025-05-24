import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

class TradingSentimentAnalyzer:
    def __init__(self):
        self.trading_data = None
        self.sentiment_data = None
        self.merged_data = None
        self.trader_performance = None
        self.insights = {}
        
    def _parse_dates_flexible(self, date_series, column_name="date"):
        """Flexibly parse dates with multiple format attempts"""
        print(f"Parsing {column_name} column...")
        
        # Try different date formats
        date_formats = [
            '%d-%m-%Y %H:%M',  # DD-MM-YYYY HH:MM
            '%m-%d-%Y %H:%M',  # MM-DD-YYYY HH:MM
            '%Y-%m-%d %H:%M',  # YYYY-MM-DD HH:MM
            '%d/%m/%Y %H:%M',  # DD/MM/YYYY HH:MM
            '%m/%d/%Y %H:%M',  # MM/DD/YYYY HH:MM
            '%Y-%m-%d',        # YYYY-MM-DD
            '%d-%m-%Y',        # DD-MM-YYYY
            '%m-%d-%Y',        # MM-DD-YYYY
        ]
        
        parsed_dates = None
        successful_format = None
        
        for fmt in date_formats:
            try:
                parsed_dates = pd.to_datetime(date_series, format=fmt, errors='coerce')
                if parsed_dates.notna().sum() > len(parsed_dates) * 0.8:  # At least 80% success rate
                    successful_format = fmt
                    print(f" Successfully parsed {column_name} using format: {fmt}")
                    break
            except:
                continue
        
        # If no format worked, try pandas' flexible parsing
        if parsed_dates is None or parsed_dates.notna().sum() < len(parsed_dates) * 0.8:
            try:
                parsed_dates = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
                print(f" Used flexible parsing for {column_name} (dayfirst=True)")
            except:
                try:
                    parsed_dates = pd.to_datetime(date_series, errors='coerce')
                    print(f" Used flexible parsing for {column_name} (default)")
                except:
                    print(f"❌ Failed to parse {column_name}")
                    return None
        
        return parsed_dates
    
    def load_data(self, trading_file_path, sentiment_file_path):
        """Load and preprocess trading and sentiment data"""
        print("Loading datasets...")
        
        # Load trading data
        try:
            self.trading_data = pd.read_csv(trading_file_path)
            print(f" Trading data loaded: {len(self.trading_data)} records")
            print(f"  Columns: {list(self.trading_data.columns)}")
        except Exception as e:
            print(f"Error loading trading data: {e}")
            return False
            
        # Load sentiment data
        try:
            self.sentiment_data = pd.read_csv(sentiment_file_path)
            print(f" Sentiment data loaded: {len(self.sentiment_data)} records")
            print(f"  Columns: {list(self.sentiment_data.columns)}")
        except Exception as e:
            print(f"Error loading sentiment data: {e}")
            return False
            
        # Preprocess data
        self._preprocess_data()
        return True
    
    def _preprocess_data(self):
        """Clean and preprocess the datasets"""
        print("Preprocessing data...")
        
        # Clean trading data - handle timestamps
        timestamp_columns = ['Timestamp IST', 'timestamp', 'Timestamp', 'Date', 'date']
        trading_timestamp_col = None
        
        for col in timestamp_columns:
            if col in self.trading_data.columns:
                trading_timestamp_col = col
                break
        
        if trading_timestamp_col:
            parsed_dates = self._parse_dates_flexible(self.trading_data[trading_timestamp_col], trading_timestamp_col)
            if parsed_dates is not None:
                self.trading_data['date'] = parsed_dates.dt.date
                self.trading_data['datetime'] = parsed_dates
            else:
                print(" Could not parse trading data timestamps")
                return False
        else:
            print(" No timestamp column found in trading data")
            return False
        
        # Standardize trading data column names
        column_mapping = {
            'Account': 'account',
            'Coin': 'coin', 
            'Execution Price': 'execution_price',
            'Size USD': 'size_usd',
            'Side': 'side',
            'Closed PnL': 'closed_pnl',
            'Size Tokens': 'size_tokens'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in self.trading_data.columns:
                self.trading_data[new_col] = self.trading_data[old_col]
        
        # Clean sentiment data - handle dates
        sentiment_date_columns = ['date', 'Date', 'timestamp', 'Timestamp']
        sentiment_date_col = None
        
        for col in sentiment_date_columns:
            if col in self.sentiment_data.columns:
                sentiment_date_col = col
                break
        
        if sentiment_date_col:
            if sentiment_date_col != 'date':
                # If it's not already 'date', try to parse it
                parsed_dates = self._parse_dates_flexible(self.sentiment_data[sentiment_date_col], sentiment_date_col)
                if parsed_dates is not None:
                    self.sentiment_data['date'] = parsed_dates.dt.date
                else:
                    print(" Could not parse sentiment data dates")
                    return False
            else:
                # It's already 'date', but might need parsing
                try:
                    parsed_dates = pd.to_datetime(self.sentiment_data['date'])
                    self.sentiment_data['date'] = parsed_dates.dt.date
                except:
                    parsed_dates = self._parse_dates_flexible(self.sentiment_data['date'], 'date')
                    if parsed_dates is not None:
                        self.sentiment_data['date'] = parsed_dates.dt.date
                    else:
                        print(" Could not parse sentiment data dates")
                        return False
        else:
            print(" No date column found in sentiment data")
            return False
        
        # Convert numeric columns
        numeric_cols = ['execution_price', 'size_usd', 'closed_pnl', 'size_tokens']
        for col in numeric_cols:
            if col in self.trading_data.columns:
                self.trading_data[col] = pd.to_numeric(self.trading_data[col], errors='coerce')
        
        # Handle sentiment value column
        value_columns = ['value', 'Value', 'score', 'Score', 'sentiment_value']
        for col in value_columns:
            if col in self.sentiment_data.columns:
                self.sentiment_data['value'] = pd.to_numeric(self.sentiment_data[col], errors='coerce')
                break
        
        # Handle sentiment classification column
        class_columns = ['classification', 'Classification', 'sentiment', 'Sentiment', 'label', 'Label']
        for col in class_columns:
            if col in self.sentiment_data.columns:
                self.sentiment_data['classification'] = self.sentiment_data[col]
                break
        
        # Remove rows with missing critical data
        initial_trading_len = len(self.trading_data)
        initial_sentiment_len = len(self.sentiment_data)
        
        self.trading_data = self.trading_data.dropna(subset=['date', 'closed_pnl'])
        self.sentiment_data = self.sentiment_data.dropna(subset=['date', 'value'])
        
        print(f"Trading data: {initial_trading_len} → {len(self.trading_data)} records (removed {initial_trading_len - len(self.trading_data)} invalid)")
        print(f"Sentiment data: {initial_sentiment_len} → {len(self.sentiment_data)} records (removed {initial_sentiment_len - len(self.sentiment_data)} invalid)")
        print("Data preprocessing completed")
        
        return True
    
    def analyze_trader_performance(self):
        """Analyze individual trader performance metrics"""
        print("Analyzing trader performance...")
        
        if 'account' not in self.trading_data.columns:
            print("No account column found for trader analysis")
            return None
        
        performance_metrics = []
        
        for account in self.trading_data['account'].unique():
            trader_data = self.trading_data[self.trading_data['account'] == account].copy()
            
            # Skip if insufficient data
            if len(trader_data) < 2:
                continue
            
            # Calculate performance metrics
            total_pnl = trader_data['closed_pnl'].sum()
            total_volume = trader_data['size_usd'].sum() if 'size_usd' in trader_data.columns else 0
            total_trades = len(trader_data)
            
            # Win rate calculation
            profitable_trades = len(trader_data[trader_data['closed_pnl'] > 0])
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Average trade metrics
            avg_trade_size = trader_data['size_usd'].mean() if 'size_usd' in trader_data.columns else 0
            avg_pnl = trader_data['closed_pnl'].mean()
            
            # Risk metrics
            pnl_std = trader_data['closed_pnl'].std()
            sharpe_ratio = avg_pnl / pnl_std if pnl_std > 0 else 0
            
            # ROI calculation
            roi = (total_pnl / total_volume) * 100 if total_volume > 0 else 0
            
            # Max drawdown (simplified)
            cumulative_pnl = trader_data['closed_pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            
            performance_metrics.append({
                'account': account,
                'total_pnl': total_pnl,
                'total_volume': total_volume,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_trade_size': avg_trade_size,
                'avg_pnl': avg_pnl,
                'roi': roi,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'pnl_volatility': pnl_std
            })
        
        self.trader_performance = pd.DataFrame(performance_metrics)
        self.trader_performance = self.trader_performance.sort_values('total_pnl', ascending=False)
        
        print(f"Analyzed {len(self.trader_performance)} traders")
        return self.trader_performance
    
    def merge_sentiment_trading_data(self):
        """Merge trading and sentiment data by date"""
        print("Merging trading and sentiment data...")
        
        # Aggregate trading data by date
        daily_trading = self.trading_data.groupby('date').agg({
            'closed_pnl': ['sum', 'mean', 'count'],
            'size_usd': ['sum', 'mean'] if 'size_usd' in self.trading_data.columns else ['count'],
            'account': 'nunique' if 'account' in self.trading_data.columns else 'count'
        }).reset_index()
        
        # Flatten column names
        if 'size_usd' in self.trading_data.columns:
            daily_trading.columns = [
                'date', 'total_pnl', 'avg_pnl', 'trade_count',
                'total_volume', 'avg_volume', 'active_traders'
            ]
        else:
            daily_trading.columns = [
                'date', 'total_pnl', 'avg_pnl', 'trade_count',
                'record_count', 'active_accounts'
            ]
            daily_trading['total_volume'] = 0
            daily_trading['avg_volume'] = 0
            daily_trading['active_traders'] = daily_trading['active_accounts']
        
        # Prepare sentiment data
        if 'classification' not in self.sentiment_data.columns:
            # Create classification based on value if it doesn't exist
            if 'value' in self.sentiment_data.columns:
                self.sentiment_data['classification'] = pd.cut(
                    self.sentiment_data['value'], 
                    bins=[0, 25, 50, 75, 100], 
                    labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed']
                )
        
        sentiment_df = self.sentiment_data[['date', 'value']].copy()
        if 'classification' in self.sentiment_data.columns:
            sentiment_df['classification'] = self.sentiment_data['classification']
        else:
            sentiment_df['classification'] = 'Unknown'
        
        # Merge data
        self.merged_data = pd.merge(daily_trading, sentiment_df, on='date', how='inner')
        
        print(f" Merged data: {len(self.merged_data)} days")
        if len(self.merged_data) == 0:
            print(" No overlapping dates found between trading and sentiment data")
            print(f"Trading date range: {self.trading_data['date'].min()} to {self.trading_data['date'].max()}")
            print(f"Sentiment date range: {self.sentiment_data['date'].min()} to {self.sentiment_data['date'].max()}")
        
        return self.merged_data
    
    def calculate_sentiment_correlations(self):
        """Calculate correlations between sentiment and trading metrics"""
        print("Calculating sentiment correlations...")
        
        if self.merged_data is None or len(self.merged_data) == 0:
            self.merge_sentiment_trading_data()
        
        if self.merged_data is None or len(self.merged_data) == 0:
            print(" No merged data available for correlation analysis")
            return {}
        
        correlations = {}
        trading_metrics = ['total_pnl', 'avg_pnl', 'trade_count', 'total_volume', 'active_traders']
        
        for metric in trading_metrics:
            if metric in self.merged_data.columns:
                try:
                    # Remove NaN values for correlation calculation
                    clean_data = self.merged_data[['value', metric]].dropna()
                    
                    if len(clean_data) > 5:  # Need at least 5 data points
                        # Pearson correlation
                        pearson_corr, pearson_p = pearsonr(clean_data['value'], clean_data[metric])
                        
                        # Spearman correlation
                        spearman_corr, spearman_p = spearmanr(clean_data['value'], clean_data[metric])
                        
                        correlations[metric] = {
                            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
                        }
                except Exception as e:
                    print(f"Error calculating correlation for {metric}: {e}")
                    continue
        
        self.insights['correlations'] = correlations
        
        # Print correlation results
        if correlations:
            print("\nSENTIMENT CORRELATIONS:")
            print("-" * 50)
            for metric, corrs in correlations.items():
                print(f"{metric.upper()}:")
                print(f"  Pearson:  {corrs['pearson']['correlation']:.3f} (p={corrs['pearson']['p_value']:.3f})")
                print(f"  Spearman: {corrs['spearman']['correlation']:.3f} (p={corrs['spearman']['p_value']:.3f})")
        else:
            print(" No correlations could be calculated")
        
        return correlations
    
    def analyze_sentiment_phases(self):
        """Analyze trading performance across different sentiment phases"""
        print("Analyzing sentiment phase performance...")
        
        if self.merged_data is None or len(self.merged_data) == 0:
            self.merge_sentiment_trading_data()
        
        if self.merged_data is None or len(self.merged_data) == 0:
            print(" No merged data available for sentiment phase analysis")
            return None
        
        # Group by sentiment classification
        try:
            sentiment_analysis = self.merged_data.groupby('classification').agg({
                'total_pnl': ['mean', 'sum', 'std'],
                'avg_pnl': 'mean',
                'trade_count': 'mean',
                'total_volume': ['mean', 'sum'],
                'active_traders': 'mean',
                'value': 'mean'
            }).round(2)
            
            # Flatten column names
            sentiment_analysis.columns = [
                'avg_daily_pnl', 'total_pnl', 'pnl_volatility',
                'avg_trade_pnl', 'avg_daily_trades', 'avg_daily_volume',
                'total_volume', 'avg_active_traders', 'avg_sentiment_value'
            ]
            
            self.insights['sentiment_phases'] = sentiment_analysis
            
            print("\n SENTIMENT PHASE ANALYSIS:")
            print("-" * 60)
            print(sentiment_analysis.to_string())
            
            return sentiment_analysis
        except Exception as e:
            print(f"Error in sentiment phase analysis: {e}")
            return None
    
    def identify_trading_clusters(self):
        """Identify trader clusters based on performance characteristics"""
        print("Identifying trader clusters...")
        
        if self.trader_performance is None:
            self.analyze_trader_performance()
        
        if self.trader_performance is None or len(self.trader_performance) < 4:
            print("❌ Insufficient trader data for clustering")
            return None
        
        # Select features for clustering
        cluster_features = ['total_pnl', 'win_rate', 'roi', 'avg_trade_size', 'total_trades']
        available_features = [col for col in cluster_features if col in self.trader_performance.columns]
        
        if len(available_features) < 3:
            print("❌ Insufficient features available for clustering")
            return None
        
        # Prepare data for clustering
        cluster_data = self.trader_performance[available_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        n_clusters = min(4, len(self.trader_performance) // 2)  # Ensure reasonable cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to trader performance data
        self.trader_performance['cluster'] = clusters
        
        # Analyze cluster characteristics
        cluster_analysis = self.trader_performance.groupby('cluster').agg({
            'total_pnl': ['mean', 'median'],
            'win_rate': 'mean',
            'roi': 'mean',
            'total_trades': 'mean',
            'avg_trade_size': 'mean'
        }).round(2)
        
        cluster_labels = {
            0: "Conservative Traders",
            1: "High-Volume Traders", 
            2: "Profitable Specialists",
            3: "Risk-Takers"
        }
        
        print(f"\n TRADER CLUSTERS:")
        print("-" * 50)
        for i in range(n_clusters):
            cluster_size = len(self.trader_performance[self.trader_performance['cluster'] == i])
            print(f"\nCluster {i} - {cluster_labels.get(i, f'Group {i}')} ({cluster_size} traders):")
            if not cluster_analysis.empty:
                print(cluster_analysis.loc[i].to_string())
        
        self.insights['clusters'] = cluster_analysis
        return cluster_analysis
    
    def generate_trading_insights(self):
        """Generate actionable trading insights"""
        print("Generating trading insights...")
        
        insights = []
        
        # Performance insights
        if self.trader_performance is not None and len(self.trader_performance) > 0:
            avg_win_rate = self.trader_performance['win_rate'].mean()
            
            if avg_win_rate > 60:
                insights.append(" High average win rate suggests good market timing across traders")
            elif avg_win_rate < 40:
                insights.append(" Low average win rate indicates challenging market conditions")
        
        # Sentiment insights
        if 'sentiment_phases' in self.insights and self.insights['sentiment_phases'] is not None:
            sentiment_phases = self.insights['sentiment_phases']
            
            if not sentiment_phases.empty:
                best_sentiment = sentiment_phases['avg_daily_pnl'].idxmax()
                worst_sentiment = sentiment_phases['avg_daily_pnl'].idxmin()
                
                insights.append(f" Best performance during: {best_sentiment} periods")
                insights.append(f" Worst performance during: {worst_sentiment} periods")
        
        # Correlation insights
        if 'correlations' in self.insights and self.insights['correlations']:
            correlations = self.insights['correlations']
            
            strong_correlations = []
            for metric, corrs in correlations.items():
                if abs(corrs['pearson']['correlation']) > 0.3:
                    direction = "positive" if corrs['pearson']['correlation'] > 0 else "negative"
                    strong_correlations.append(f"{metric} ({direction})")
            
            if strong_correlations:
                insights.append(f"🔗 Strong sentiment correlations with: {', '.join(strong_correlations)}")
        
        # Risk insights
        if self.trader_performance is not None and 'sharpe_ratio' in self.trader_performance.columns:
            high_sharpe_traders = self.trader_performance[self.trader_performance['sharpe_ratio'] > 1]
            if len(high_sharpe_traders) > 0:
                insights.append(f" {len(high_sharpe_traders)} traders show excellent risk-adjusted returns")
        
        if not insights:
            insights.append(" Analysis completed with limited data - consider expanding dataset for more insights")
        
        self.insights['trading_insights'] = insights
        
        print("\n KEY INSIGHTS:")
        print("-" * 40)
        for insight in insights:
            print(f"  {insight}")
        
        return insights
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        try:
            # Set up the plot style
            plt.style.use('default')  # Use default instead of seaborn-v0_8 for compatibility
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Trader Performance Distribution
            if self.trader_performance is not None and len(self.trader_performance) > 0:
                plt.subplot(3, 3, 1)
                plt.hist(self.trader_performance['total_pnl'], bins=min(30, len(self.trader_performance)), 
                        alpha=0.7, color='skyblue')
                plt.title('Distribution of Trader PnL')
                plt.xlabel('Total PnL ($)')
                plt.ylabel('Number of Traders')
            
            # 2. Win Rate vs ROI Scatter
            if self.trader_performance is not None and len(self.trader_performance) > 0:
                plt.subplot(3, 3, 2)
                scatter = plt.scatter(self.trader_performance['win_rate'], self.trader_performance['roi'], 
                                    alpha=0.6, c=self.trader_performance['total_trades'], cmap='viridis')
                plt.xlabel('Win Rate (%)')
                plt.ylabel('ROI (%)')
                plt.title('Win Rate vs ROI (Color = Trade Count)')
                plt.colorbar(scatter, label='Total Trades')
            
            # 3. Sentiment vs Daily PnL
            if self.merged_data is not None and len(self.merged_data) > 0:
                plt.subplot(3, 3, 3)
                plt.scatter(self.merged_data['value'], self.merged_data['total_pnl'], alpha=0.6)
                plt.xlabel('Fear & Greed Index')
                plt.ylabel('Daily Total PnL ($)')
                plt.title('Sentiment vs Daily PnL')
                
                # Add correlation line
                if len(self.merged_data) > 1:
                    z = np.polyfit(self.merged_data['value'], self.merged_data['total_pnl'], 1)
                    p = np.poly1d(z)
                    plt.plot(self.merged_data['value'], p(self.merged_data['value']), "r--", alpha=0.8)
            
            # 4. Top Performers Bar Chart
            if self.trader_performance is not None and len(self.trader_performance) > 0:
                plt.subplot(3, 3, 4)
                top_n = min(10, len(self.trader_performance))
                top_performers = self.trader_performance.head(top_n)
                plt.barh(range(len(top_performers)), top_performers['total_pnl'])
                plt.yticks(range(len(top_performers)), [f"Trader {i+1}" for i in range(len(top_performers))])
                plt.xlabel('Total PnL ($)')
                plt.title(f'Top {top_n} Performers')
            
            # 5. Sentiment Distribution
            if self.sentiment_data is not None and 'classification' in self.sentiment_data.columns:
                plt.subplot(3, 3, 5)
                sentiment_counts = self.sentiment_data['classification'].value_counts()
                if len(sentiment_counts) > 0:
                    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
                    plt.title('Market Sentiment Distribution')
            
            # 6. Trading Volume Over Time
            if self.merged_data is not None and len(self.merged_data) > 0:
                plt.subplot(3, 3, 6)
                plt.plot(range(len(self.merged_data)), self.merged_data['total_volume'])
                plt.xlabel('Days')
                plt.ylabel('Daily Volume ($)')
                plt.title('Trading Volume Over Time')
            
            # 7. Correlation Heatmap
            if self.merged_data is not None and len(self.merged_data) > 1:
                plt.subplot(3, 3, 7)
                numeric_cols = ['value', 'total_pnl', 'trade_count', 'total_volume', 'active_traders']
                available_cols = [col for col in numeric_cols if col in self.merged_data.columns]
                
                if len(available_cols) > 1:
                    corr_matrix = self.merged_data[available_cols].corr()
                    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    plt.colorbar(im)
                    plt.xticks(range(len(available_cols)), available_cols, rotation=45)
                    plt.yticks(range(len(available_cols)), available_cols)
                    plt.title('Correlation Matrix')
            
            # 8. Risk-Return Scatter
            if self.trader_performance is not None and len(self.trader_performance) > 0:
                plt.subplot(3, 3, 8)
                scatter = plt.scatter(self.trader_performance['pnl_volatility'], 
                                    self.trader_performance['total_pnl'], 
                                    alpha=0.6, c=self.trader_performance['win_rate'], cmap='RdYlGn')
                plt.xlabel('PnL Volatility')
                plt.ylabel('Total PnL ($)')
                plt.title('Risk vs Return (Color = Win Rate)')
                plt.colorbar(scatter, label='Win Rate (%)')
            
            # 9. Sentiment Phases Performance
            if 'sentiment_phases' in self.insights and self.insights['sentiment_phases'] is not None:
                plt.subplot(3, 3, 9)
                sentiment_phases = self.insights['sentiment_phases']
                if not sentiment_phases.empty:
                    plt.bar(range(len(sentiment_phases)), sentiment_phases['avg_daily_pnl'])
                    plt.xticks(range(len(sentiment_phases)), sentiment_phases.index, rotation=45)
                    plt.xlabel('Sentiment Phase')
                    plt.ylabel('Average Daily PnL ($)')
                    plt.title('Performance by Sentiment Phase')
            
            plt.tight_layout()
            plt.savefig('trading_sentiment_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(" Visualizations saved as 'trading_sentiment_analysis.png'")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*80)
        print(" TRADING PERFORMANCE vs MARKET SENTIMENT ANALYSIS REPORT")
        print("="*80)
        
        # Dataset Overview
        print(f"\n DATASET OVERVIEW:")
        if self.trading_data is not None:
            print(f"  • Total Trades: {len(self.trading_data):,}")
            print(f"  • Date Range: {self.trading_data['date'].min()} to {self.trading_data['date'].max()}")
            if 'size_usd' in self.trading_data.columns:
                print(f"  • Total Volume: ${self.trading_data['size_usd'].sum():,.2f}")
            print(f"  • Total PnL: ${self.trading_data['closed_pnl'].sum():,.2f}")
        
        if self.trader_performance is not None:
            print(f"  • Total Traders: {len(self.trader_performance)}")
        
        if self.sentiment_data is not None:
            print(f"  • Sentiment Records: {len(self.sentiment_data):,}")
        
        if self.merged_data is not None:
            print(f"  • Overlapping Days: {len(self.merged_data)}")
        
        # Top Performers
        if self.trader_performance is not None and len(self.trader_performance) > 0:
            print(f"\n TOP 5 PERFORMERS:")
            top_5 = self.trader_performance.head(5)
            for i, (_, trader) in enumerate(top_5.iterrows(), 1):
                account_display = str(trader['account'])[:42] + "..." if len(str(trader['account'])) > 42 else str(trader['account'])
                print(f"  {i}. {account_display} - PnL: ${trader['total_pnl']:,.2f} - Win Rate: {trader['win_rate']:.1f}%")
        
        # Performance Summary
        if self.trader_performance is not None and len(self.trader_performance) > 0:
            print(f"\n PERFORMANCE SUMMARY:")
            print(f"  • Average Win Rate: {self.trader_performance['win_rate'].mean():.1f}%")
            print(f"  • Average ROI: {self.trader_performance['roi'].mean():.1f}%")
            print(f"  • Profitable Traders: {len(self.trader_performance[self.trader_performance['total_pnl'] > 0])}")
            print(f"  • Loss-making Traders: {len(self.trader_performance[self.trader_performance['total_pnl'] < 0])}")
        
        # Display insights
        if 'trading_insights' in self.insights:
            print(f"\n KEY INSIGHTS:")
            for insight in self.insights['trading_insights']:
                print(f"  {insight}")
        
        # Recommendations
        print(f"\n TRADING RECOMMENDATIONS:")
        print(f"  • Monitor sentiment during extreme fear periods for buying opportunities")
        print(f"  • Implement position sizing based on sentiment phases")
        print(f"  • Focus on risk management during high volatility periods")
        print(f"  • Consider contrarian strategies during extreme sentiment readings")
        
        print(f"\n Analysis Complete! Check the visualizations for detailed insights.")
        print("="*80)
    
    def run_complete_analysis(self, trading_file, sentiment_file):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive trading sentiment analysis...\n")
        
        # Load data
        if not self.load_data(trading_file, sentiment_file):
            return False
        
        # Run all analyses
        try:
            self.analyze_trader_performance()
            self.merge_sentiment_trading_data()
            self.calculate_sentiment_correlations()
            self.analyze_sentiment_phases()
            self.identify_trading_clusters()
            self.generate_trading_insights()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate final report
            self.generate_report()
            
            return True
        except Exception as e:
            print(f" Error during analysis: {e}")
            return False

# Usage Example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TradingSentimentAnalyzer()
    
    # File paths (update these with your actual file paths)
    trading_file = "historical_data.csv"
    sentiment_file = "fear_greed_index.csv"
    
    # Run complete analysis
    success = analyzer.run_complete_analysis(trading_file, sentiment_file)
    
    if not success:
        print("\n Analysis failed. Please check your data files and try again.")
        print("Make sure your files contain the expected columns:")
        print("Trading data: timestamp column, account/trader info, PnL data")
        print("Sentiment data: date column, sentiment value/score")
    
    # You can also run individual analyses:
    # analyzer.load_data(trading_file, sentiment_file)
    # analyzer.analyze_trader_performance()
    # analyzer.calculate_sentiment_correlations()
    # etc.
