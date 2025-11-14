from config import Config

c = Config()
print('=== 交易配置 ===')
print('交易品种:')
for i, s in enumerate(c.symbol_list):
    print(f'  {i+1}. {s}')

print(f'\n仓位管理:')
print(f'  模式: {c.position_sizing_mode}')
print(f'  仓位比例: {c.position_size_pct*100}%')

print(f'\n止盈止损 (价格变动):')
print(f'  止损: {c.scalping_stop_loss_pct}%')
print(f'  止盈: {c.scalping_take_profit_pct}%')

print(f'\n回撤保护:')
print(f'  当日限制: {c.daily_drawdown_limit*100}%')
print(f'  总回撤限制: {c.overall_drawdown_limit*100}%')
print(f'  Kill Switch触发线: ${c.initial_capital * (1.0 - c.overall_drawdown_limit):.2f}')
