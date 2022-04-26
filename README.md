STABLE BASELINES STOCK/CRYPTO TRADER

A trading bot using
    - InvestPy to gather data
    - FinTa to add technical indicators
    - StableBaselines3 to learn and make trades

    
    - Use technical indicators to inform the bot X
    - Pass in crypto held, balance and net worth to the bot | Didnt' work
    - Use other stock prices to inform the bot
        - Gold X --
        - Oil
        - S&P 500
        - Silver
        - Keep these parametized so I can add them to data easily
 
        
    - SB Params:
        - Add callbacks to stop at explained variance = 1
        - Search different policies and different models in SB
        
    - Try trading different stocks
        - parametise so I can train mutliple stocks during the night
        - Top 5-10 stocks on ASX200
            - Get with yfinance if invest.py isn't working
        
    - Train each week, if avg of all training sessions > previous avg: replace model