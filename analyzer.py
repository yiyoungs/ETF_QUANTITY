import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from config import StrategyConfig
import logging
import os

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
        
        # 回撤恢复天数：记录每次从回撤恢复到前高所需的天数
        recovery_days_list = []
        peak_nav = df['nav'].iloc[0]
        peak_date = df['date'].iloc[0]
        in_drawdown = False
        drawdown_start_date = None

        for i in range(1, len(df)):
            current_nav = df['nav'].iloc[i]
            current_date = df['date'].iloc[i]

            if current_nav > peak_nav:
                # 创新高
                if in_drawdown:
                    # 从回撤中恢复到前高，记录恢复天数
                    recovery_days = (current_date - drawdown_start_date).days
                    recovery_days_list.append(recovery_days)
                    in_drawdown = False
                peak_nav = current_nav
                peak_date = current_date
            elif not in_drawdown and current_nav < peak_nav:
                # 从峰值开始回撤
                in_drawdown = True
                drawdown_start_date = peak_date

        max_recovery_days = max(recovery_days_list) if recovery_days_list else 0
        avg_recovery_days = sum(recovery_days_list) / len(recovery_days_list) if recovery_days_list else 0
        
        self._calculate_exposure_metrics(df)
        
        self.metrics.update({
            'total_return': total_return,
            'annualized_return': annual_return / 100,
            'annualized_volatility': annual_std / 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown / 100,
            'win_rate': win_rate / 100,
            'profit_factor': profit_factor,
            'max_recovery_days': max_recovery_days,
            'avg_recovery_days': avg_recovery_days,
            'total_days': total_days,
            'years': years
        })
        
        return self.metrics
    
    def _calculate_exposure_metrics(self, df):
        """
        计算持仓暴露度指标
        - 风险资产仓位占比：持有非国债 ETF 的天数 / 总交易日天数
        - 平均持仓数量：平均每天持有几只风险 ETF
        - 最大连续空仓天数：连续全仓国债的最长天数
        """
        cash_etf = StrategyConfig.CASH_ETF_CODE
        
        def has_risk_assets(positions):
            if not positions:
                return False
            for code in positions.keys():
                if code != cash_etf and positions[code] > 0:
                    return True
            return False
        
        def count_risk_assets(positions):
            count = 0
            if positions:
                for code, shares in positions.items():
                    if code != cash_etf and shares > 0:
                        count += 1
            return count
        
        df['has_risk_assets'] = df['positions'].apply(has_risk_assets)
        df['risk_asset_count'] = df['positions'].apply(count_risk_assets)
        
        risk_days = df['has_risk_assets'].sum()
        risk_exposure_ratio = risk_days / len(df)
        
        avg_position_count = df['risk_asset_count'].mean()
        
        df['is_all_cash'] = ~df['has_risk_assets']
        df['cash_streak'] = df['is_all_cash'].astype(int).groupby(
            (df['is_all_cash'] != df['is_all_cash'].shift()).cumsum()
        ).cumsum()
        max_cash_streak = df['cash_streak'].max()
        
        self.metrics.update({
            'risk_exposure_ratio': risk_exposure_ratio,
            'avg_position_count': avg_position_count,
            'max_cash_streak': int(max_cash_streak)
        })
    
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
            f.write("\n")
            f.write("【持仓暴露度分析】\n")
            f.write(f"  风险资产仓位占比: {self.metrics['risk_exposure_ratio'] * 100:.2f}%\n")
            f.write(f"  平均持仓数量: {self.metrics['avg_position_count']:.1f} 只\n")
            f.write(f"  最大连续空仓天数: {self.metrics['max_cash_streak']} 天\n")
        
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
        print(f"  年化收益率: {self.metrics['annualized_return'] * 100:.2f}%")
        print(f"  年化波动率: {self.metrics['annualized_volatility'] * 100:.2f}%")
        print(f"  夏普比率: {self.metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {self.metrics['max_drawdown'] * 100:.2f}%")
        print(f"  胜率: {self.metrics['win_rate'] * 100:.2f}%")
        print(f"  盈亏比: {self.metrics['profit_factor']:.2f}")
        print(f"\n【持仓暴露度】")
        print(f"  风险资产仓位占比: {self.metrics['risk_exposure_ratio'] * 100:.2f}%")
        print(f"  平均持仓数量: {self.metrics['avg_position_count']:.1f} 只")
        print(f"  最大连续空仓天数: {self.metrics['max_cash_streak']} 天")

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