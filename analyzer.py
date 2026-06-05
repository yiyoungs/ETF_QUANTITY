import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from config import StrategyConfig
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'analyzer.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Analyzer:
    def __init__(self, results_df):
        self.results_df = results_df
        self.metrics = {}
    
    def calculate_metrics(self):
        df = self.results_df.copy()
        
        total_return = (df['nav'].iloc[-1] - 1) * 100
        
        years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        
        daily_std = df['daily_return'].std() * 100
        annual_std = daily_std * np.sqrt(252)
        
        sharpe_ratio = (annual_return / 100) / (annual_std / 100)
        
        df['cummax'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min() * 100
        
        winning_days = df[df['daily_return'] > 0].shape[0]
        total_days = df.shape[0]
        win_rate = (winning_days / total_days) * 100
        
        avg_win = df[df['daily_return'] > 0]['daily_return'].mean() * 100
        avg_loss = df[df['daily_return'] < 0]['daily_return'].mean() * 100
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        
        df['equity'] = df['nav'].cummax()
        df['watermark'] = (df['nav'] >= df['equity'].shift(1)).cumsum()
        recovery_days = df[df['watermark'] != df['watermark'].shift(1)].shape[0] - 1
        
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annual_return / 100,
            'annualized_volatility': annual_std / 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown / 100,
            'win_rate': win_rate / 100,
            'profit_factor': profit_factor,
            'recovery_days': recovery_days,
            'total_days': total_days,
            'years': years
        }
        
        return self.metrics
    
    def generate_report(self, output_dir='reports'):
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.metrics:
            self.calculate_metrics()
        
        report_path = os.path.join(output_dir, 'performance_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("          ETF 双重动量策略绩效报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"回测时间范围: {self.results_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {self.results_df['date'].iloc[-1].strftime('%Y-%m-%d')}\n")
            f.write(f"初始资金: {StrategyConfig.INITIAL_CAPITAL:,} 元\n")
            f.write(f"交易成本: {StrategyConfig.TRANSACTION_COST * 100:.2f}% (单边)\n")
            f.write("\n")
            
            f.write("【收益指标】\n")
            f.write(f"  总收益率: {self.metrics['total_return']:.2f}%\n")
            f.write(f"  年化收益率: {self.metrics['annualized_return'] * 100:.2f}%\n")
            f.write(f"  年化波动率: {self.metrics['annualized_volatility'] * 100:.2f}%\n")
            f.write("\n")
            
            f.write("【风险指标】\n")
            f.write(f"  夏普比率: {self.metrics['sharpe_ratio']:.2f}\n")
            f.write(f"  最大回撤: {self.metrics['max_drawdown'] * 100:.2f}%\n")
            f.write("\n")
            
            f.write("【交易统计】\n")
            f.write(f"  总交易天数: {self.metrics['total_days']} 天\n")
            f.write(f"  回测年限: {self.metrics['years']:.2f} 年\n")
            f.write(f"  胜率: {self.metrics['win_rate'] * 100:.2f}%\n")
            f.write(f"  盈亏比: {self.metrics['profit_factor']:.2f}\n")
        
        logger.info(f"绩效报告已保存至: {report_path}")
        
        self.plot_nav(output_dir)
        self.plot_drawdown(output_dir)
        
        return self.metrics
    
    def plot_nav(self, output_dir):
        df = self.results_df.copy()
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['nav'], label='策略净值', linewidth=2)
        if 'benchmark' in df.columns:
            plt.plot(df['date'], df['benchmark'], label='沪深300基准', linewidth=2, alpha=0.7)
        
        plt.title('ETF 双重动量策略净值曲线', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'nav_curve.png'), dpi=150)
        plt.close()
        
        logger.info(f"净值曲线已保存至: {os.path.join(output_dir, 'nav_curve.png')}")
    
    def plot_drawdown(self, output_dir):
        df = self.results_df.copy()
        df['cummax'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df['date'], df['drawdown'], 0, where=df['drawdown'] < 0, 
                         color='red', alpha=0.3)
        plt.plot(df['date'], df['drawdown'], label='回撤', color='red', linewidth=2)
        
        plt.title('ETF 双重动量策略回撤曲线', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('回撤', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'drawdown_curve.png'), dpi=150)
        plt.close()
        
        logger.info(f"回撤曲线已保存至: {os.path.join(output_dir, 'drawdown_curve.png')}")
    
    def print_summary(self):
        if not self.metrics:
            self.calculate_metrics()
        
        print("=" * 60)
        print("          ETF 双重动量策略绩效摘要")
        print("=" * 60)
        print(f"\n回测时间: {self.results_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {self.results_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"初始资金: {StrategyConfig.INITIAL_CAPITAL:,} 元")
        print(f"\n【核心指标】")
        print(f"  总收益率: {self.metrics['total_return']:.2f}%")
        print(f"  年化收益率: {self.metrics['annual_return']:.2f}%")
        print(f"  年化波动率: {self.metrics['annual_std']:.2f}%")
        print(f"  夏普比率: {self.metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {self.metrics['max_drawdown']:.2f}%")
        print(f"  胜率: {self.metrics['win_rate']:.2f}%")
        print(f"  盈亏比: {self.metrics['profit_factor']:.2f}")

if __name__ == '__main__':
    logger.info("=== 绩效分析模块测试 ===")
    
    test_df = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=500, freq='B'),
        'nav': np.cumprod(1 + np.random.normal(0.0005, 0.01, 500)),
        'daily_return': np.random.normal(0.0005, 0.01, 500)
    })
    
    analyzer = Analyzer(test_df)
    metrics = analyzer.calculate_metrics()
    
    logger.info(f"计算的指标:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    analyzer.print_summary()
    analyzer.generate_report()
    
    logger.info("=== 测试完成 ===")