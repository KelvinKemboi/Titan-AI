import ta
import yfinance as yf

from titan.config import WEIGHTS
class RoboAnalyst:
    def __init__(self, ticker):
        self.ticker = ticker
        self.score = 0
        self.metrics = {}
        self.memo = ""
        self.valid = False

    def analyze(self):
        """
        Performs a full quantitative audit on the ticker.
        Returns True if successful, False if data is insufficient/corrupt.
        """
        try:
            stock = yf.Ticker(self.ticker)

            # 1. Fetch History (Silent Fail Protection)
            # We ask for 2y to ensure 200-day Moving Average can be calculated
            hist = stock.history(period="2y")
            if hist.empty or len(hist) < 200:
                return False

            # 2. Fetch Fundamentals
            info = stock.info

            # QUANTITATIVE ANALYSIS
            close = hist['Close']

            # A. Technical Factors
            current_price = close.iloc[-1]
            rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1] # relative strength index
            macd = ta.trend.MACD(close).macd_diff().iloc[-1] # moving average convergence divergence(used to spot price trends, measure market momentum, and find buy or sell signals)
            sma_200 = close.rolling(window=200).mean().iloc[-1] # 200-day simple moving average
            trend = "Bullish" if current_price > sma_200 else "Bearish"

            # B. Fundamental Factors (with defaults for missing data)
            peg = info.get('pegRatio') # price/earnings to growth ratio
            pe = info.get('trailingPE') # price/earnings ratio

            # Smart Valuation Logic
            if peg is not None:
                val_metric = peg
                val_type = "PEG"
            elif pe is not None:
                val_metric = pe / 25  # Normalize PE to PEG scale
                val_type = "P/E"
            else:
                val_metric = 5.0  # Assume expensive
                val_type = "Unknown"

            margins = info.get('profitMargins', 0.0)
            debt_eq = info.get('debtToEquity', 100)
            beta = info.get('beta', 1.0)

            # SCORING ALGORITHM (0-100)

            # Value: PEG < 1.0 is elite (100). PEG > 3.0 is poor (0).
            val_score = max(0, min(100, (3.0 - val_metric) * 50))

            # Momentum(speed and strength of the trend): We want RSI 50-70.
            # Penalize if Overbought (>75) or Oversold (<30)
            if 40 <= rsi <= 75:
                mom_score = 100
            else:
                mom_score = 50
            if macd > 0:
                mom_score += 10  # Bonus for rising MACD
            if trend == "Bearish":
                mom_score -= 30  # Penalty for downtrend
            mom_score = max(0, min(100, mom_score)) 

            # Quality(how valuable it is): Margins > 20% is elite.
            qual_score = max(0, min(100, margins * 500))

            # Solvency(how healthy it is): Debt/Equity < 50% is elite.
            if debt_eq is None:
                debt_eq = 100
            solv_score = max(0, min(100, (200 - debt_eq) * 0.5))

            # Volatility(how risky it is): Beta < 1.0 is safe.
            if beta is None:
                beta = 1.0
            vol_score = max(0, min(100, (1.8 - beta) * 100))

            # Final Weighted Score
            self.score = (
                val_score * WEIGHTS['Value'] +
                mom_score * WEIGHTS['Momentum'] +
                qual_score * WEIGHTS['Quality'] +
                solv_score * WEIGHTS['Solvency'] +
                vol_score * WEIGHTS['Volatility']
            )

            self.metrics = {
                'Price': current_price, 'RSI': rsi, 'Trend': trend,
                'Val_Metric': val_metric, 'Val_Type': val_type,
                'Margin': margins, 'Debt': debt_eq, 'Beta': beta,
                'Scores': [val_score, mom_score, qual_score, solv_score, vol_score],
            }
            self.valid = True
            return True

        except Exception:
            return False

    def generate_memo(self):
        """Generates the Wall Street style write-up."""
        m = self.metrics
        s = self.score

        if s >= 85:
            rating, color = "STRONG BUY", "green"
        elif s >= 65:
            rating, color = "BUY", "blue"
        elif s >= 45:
            rating, color = "HOLD", "orange"
        else:
            rating, color = "SELL", "red"

        self.memo = f"""
        ####Rating: :{color}[{rating}] (Score: {int(s)})

        **Investment Thesis:**
        {self.ticker} is currently trading at **${m['Price']:.2f}**.
        The model has flagged this asset based on a **{m['Val_Type']} of {m['Val_Metric']:.2f}** and a **{m['Trend']}** long-term trend profile.

        **Key Drivers:**
        * **Momentum:** RSI is {m['RSI']:.1f}. {' Healthy buying pressure.' if 40 < m['RSI'] < 70 else 'Caution: Potential reversal zone.'}
        * **Quality:** Net Margins of {m['Margin']:.1%} suggest {'heavy competitive moat.' if m['Margin'] > 0.20 else 'standard industry profitability.'}
        * **Risk:** Beta of {m['Beta']:.2f} indicates {'low volatility.' if m['Beta'] < 1.0 else 'higher than average market sensitivity.'}
        """
