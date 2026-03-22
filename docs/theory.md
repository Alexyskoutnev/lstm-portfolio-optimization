# Theory Guide: Risk-Adjusted Portfolio Optimization

A comprehensive guide to the financial and mathematical concepts behind this project. Written for someone with a math/ML background but limited finance experience.

---

## Table of Contents

1. [What Is a Portfolio?](#1-what-is-a-portfolio)
2. [Returns: How We Measure Gains and Losses](#2-returns)
3. [Risk: Volatility and Beyond](#3-risk)
4. [Modern Portfolio Theory (Markowitz)](#4-modern-portfolio-theory)
5. [The Efficient Frontier](#5-the-efficient-frontier)
6. [Risk-Adjusted Performance Metrics](#6-risk-adjusted-performance-metrics)
7. [Downside Risk Metrics](#7-downside-risk-metrics)
8. [Volatility Regimes](#8-volatility-regimes)
9. [Backtesting](#9-backtesting)
10. [LSTM and Deep Learning for Finance](#10-lstm-and-deep-learning-for-finance)
11. [Glossary](#11-glossary)

---

## 1. What Is a Portfolio?

A **portfolio** is simply a collection of financial assets (stocks, bonds, ETFs, etc.) held by an investor. The key question in portfolio management is:

> **How should I divide my money across different assets?**

For example, if you have $10,000 and 3 ETFs, you might put:
- 50% in XLK (Technology) → $5,000
- 30% in XLF (Financials) → $3,000
- 20% in XLE (Energy) → $2,000

These percentages are called **weights** (denoted **w**). Weights must sum to 1 (or 100%).

### What Are ETFs?

An **ETF (Exchange-Traded Fund)** is a basket of stocks bundled together and traded like a single stock. In our project, we use **sector ETFs** — each one tracks all the major companies in one industry sector of the S&P 500:

| ETF  | Sector                | Example Companies |
|------|-----------------------|-------------------|
| XLK  | Technology            | Apple, Microsoft, NVIDIA |
| XLF  | Financials            | JPMorgan, Berkshire, Visa |
| XLE  | Energy                | ExxonMobil, Chevron |
| XLY  | Consumer Discretionary| Amazon, Tesla, McDonald's |
| XLV  | Health Care           | UnitedHealth, J&J, Pfizer |
| XLP  | Consumer Staples      | Procter & Gamble, Coca-Cola |
| XLI  | Industrials           | Caterpillar, UPS, Boeing |
| XLB  | Materials             | Linde, Sherwin-Williams |
| XLRE | Real Estate           | Prologis, American Tower |
| XLU  | Utilities             | NextEra Energy, Duke Energy |
| XLC  | Communication Services| Meta, Alphabet, Netflix |

Why sector ETFs instead of individual stocks? They provide **diversification within each sector** automatically, and reduce noise from individual company events.

### Adjusted Close Price

When we download price data, we use the **Adjusted Close** price. This is the closing price adjusted for:
- **Dividends**: Cash payments companies make to shareholders. If a $100 stock pays a $2 dividend, the raw price drops to $98, but the adjusted price accounts for the value you received.
- **Stock Splits**: When a company splits its shares (e.g., 2-for-1), the raw price halves, but the adjusted price stays smooth.

Using adjusted close ensures our return calculations reflect the **total return** an investor actually experienced.

---

## 2. Returns

**Returns** measure how much an investment gained or lost over a period. There are two types we use:

### Simple (Arithmetic) Returns

The straightforward percentage change:

$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$

Where:
- $P_t$ = price today
- $P_{t-1}$ = price yesterday

**Example**: Price goes from $100 to $110 → return = (110 - 100) / 100 = **10%** or 0.10

**Advantage**: Easy to interpret. A 5% return on $1000 = $50 gain.

**Disadvantage**: Not additive over time. If you gain 10% then lose 10%, you DON'T end up where you started:
- $100 × 1.10 = $110
- $110 × 0.90 = $99 ← you lost $1!

### Log (Logarithmic) Returns

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$

**Example**: Price goes from $100 to $110 → log return = ln(110/100) = **0.0953** (≈ 9.53%)

**Key advantage**: Log returns are **additive over time**. The multi-day log return is just the sum of daily log returns:

$$r_{t_1 \to t_n} = r_{t_1} + r_{t_2} + \cdots + r_{t_n}$$

This property makes log returns mathematically cleaner for:
- Statistical modeling (they're closer to normally distributed)
- Portfolio optimization
- Cumulative return computation

**In our project**: We compute log returns for all modeling and analysis, and simple returns where needed for interpretation.

### Annualized Returns

Daily returns are tiny numbers (fractions of a percent). To make them interpretable, we **annualize** them:

$$\text{Annualized Return} = \bar{r}_{\text{daily}} \times 252$$

Where 252 is the approximate number of trading days in a year (weekends and holidays excluded).

---

## 3. Risk

In finance, **risk = uncertainty about future returns**. The primary measure is:

### Volatility (Standard Deviation)

$$\sigma = \text{std}(r_1, r_2, \ldots, r_T)$$

Annualized:

$$\sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}$$

**Why √252 and not 252?** Because variance scales linearly with time (if returns are independent), and standard deviation is the square root of variance:

$$\text{Var}_{annual} = 252 \times \text{Var}_{daily} \implies \sigma_{annual} = \sqrt{252} \times \sigma_{daily}$$

**Example**: A daily std of 1% → annual volatility ≈ 1% × √252 ≈ **15.87%**

### Covariance and Correlation

When combining assets into a portfolio, we need to know how they **move together**:

**Covariance** ($\sigma_{ij}$): Measures the joint variability of two assets.
- Positive: they tend to move in the same direction
- Negative: they tend to move in opposite directions
- Zero: no linear relationship

**Correlation** ($\rho_{ij}$): Normalized covariance, bounded between -1 and +1:

$$\rho_{ij} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}$$

**The covariance matrix** ($\Sigma$) is an N×N matrix where:
- Diagonal entries = variance of each asset
- Off-diagonal entries = covariance between pairs

$$\Sigma = \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \cdots \\ \sigma_{21} & \sigma_2^2 & \cdots \\ \vdots & & \ddots \end{bmatrix}$$

### Why Diversification Works

If two assets aren't perfectly correlated ($\rho < 1$), combining them reduces portfolio volatility. This is the fundamental insight behind portfolio theory.

**Example**: Tech stocks and utility stocks often have low correlation. When tech crashes (2000 dot-com), utilities may be stable, and vice versa. Holding both smooths out your returns.

---

## 4. Modern Portfolio Theory (Markowitz)

Harry Markowitz (1952) formalized portfolio selection as an **optimization problem**. His key insight:

> **Don't just pick the highest-return assets. Optimize the TRADE-OFF between return and risk.**

### The Math

Given N assets with:
- **Expected returns**: $\mu = [\mu_1, \mu_2, \ldots, \mu_N]^T$ (vector)
- **Covariance matrix**: $\Sigma$ (N×N matrix)
- **Portfolio weights**: $w = [w_1, w_2, \ldots, w_N]^T$ (what we're solving for)

**Portfolio expected return**:

$$R_p = w^T \mu = \sum_{i=1}^{N} w_i \mu_i$$

**Portfolio variance**:

$$\sigma_p^2 = w^T \Sigma w = \sum_{i=1}^{N} \sum_{j=1}^{N} w_i w_j \sigma_{ij}$$

**Portfolio volatility** (standard deviation):

$$\sigma_p = \sqrt{w^T \Sigma w}$$

### Minimum Variance Portfolio

The simplest optimization: find weights that minimize risk, regardless of return.

$$\min_w \quad w^T \Sigma w$$
$$\text{subject to:} \quad \sum_i w_i = 1, \quad w_i \geq 0 \text{ (long-only)}$$

This is a **quadratic program** (quadratic objective, linear constraints), which has a unique global solution.

### Maximum Sharpe Ratio Portfolio

Find weights that maximize **risk-adjusted return** (more on Sharpe ratio below):

$$\max_w \quad \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$
$$\text{subject to:} \quad \sum_i w_i = 1, \quad w_i \geq 0$$

Where $r_f$ is the risk-free rate (typically the US Treasury rate; we use 0 for simplicity).

This is trickier — it's a **fractional program** (ratio of linear to square root). In practice, we minimize the negative Sharpe ratio using scipy's SLSQP optimizer.

### Limitations of Markowitz MVO

1. **Estimation error**: Small changes in estimated $\mu$ and $\Sigma$ → large changes in optimal weights. The model is very sensitive to input parameters.
2. **Assumes normal distribution**: Real returns have fat tails (extreme events are more common than a normal distribution predicts).
3. **Static parameters**: Assumes $\mu$ and $\Sigma$ are constant over time. In reality, market dynamics shift.
4. **Linear relationships**: Only captures linear correlations. Nonlinear dependencies (e.g., assets that are uncorrelated normally but crash together in crises) are missed.

These limitations motivate exploring ML-based approaches (LSTM).

---

## 5. The Efficient Frontier

The **efficient frontier** is the set of all portfolios that offer the **highest return for each level of risk** (or equivalently, the lowest risk for each level of return).

### How It's Computed

For a range of target returns $R^*$, solve:

$$\min_w \quad w^T \Sigma w$$
$$\text{subject to:} \quad w^T \mu = R^*, \quad \sum_i w_i = 1, \quad w_i \geq 0$$

This traces out a curve in (volatility, return) space.

### Key Points on the Frontier

```
Return ↑
       |           * Max Sharpe (tangent portfolio)
       |         /
       |       /  ← Efficient Frontier
       |     /
       |   * Min Variance
       |  /
       | /
       |/
       +------------------→ Volatility
```

- **Min Variance Portfolio**: Leftmost point. Lowest possible risk.
- **Max Sharpe Portfolio**: The point where a line from the risk-free rate is tangent to the frontier. Best risk-adjusted return.
- **Portfolios BELOW the frontier**: Suboptimal. You can get higher return for the same risk.
- **Portfolios ABOVE the frontier**: Impossible with these assets.

### Reading the Efficient Frontier Plot

In our generated `efficient_frontier.png`:
- The **color** of each point represents its Sharpe ratio (yellow = high, purple = low)
- The **red star** = Max Sharpe portfolio
- The **blue star** = Min Variance portfolio
- The curve bends because diversification reduces risk more than it reduces return

---

## 6. Risk-Adjusted Performance Metrics

Raw return alone is misleading. A 20% return with 50% volatility is worse than 15% return with 10% volatility. Risk-adjusted metrics account for this.

### Sharpe Ratio

The most widely used risk-adjusted metric. Invented by William Sharpe (1966, refined 1994).

$$\text{Sharpe} = \frac{R_p - r_f}{\sigma_p}$$

Where:
- $R_p$ = annualized portfolio return
- $r_f$ = risk-free rate (e.g., Treasury bill yield)
- $\sigma_p$ = annualized portfolio volatility

**Interpretation**: Return earned per unit of total risk.

| Sharpe Ratio | Interpretation |
|-------------|----------------|
| < 0 | Losing money or worse than risk-free |
| 0 – 0.5 | Poor to below average |
| 0.5 – 1.0 | Acceptable |
| 1.0 – 2.0 | Good |
| > 2.0 | Excellent (rare for long-term strategies) |

**Limitation**: Sharpe treats upside and downside volatility equally. But investors don't mind upside volatility (big gains)! They care about downside volatility (big losses).

### Sortino Ratio

Addresses Sharpe's limitation by only penalizing **downside** volatility.

$$\text{Sortino} = \frac{R_p - \text{MAR}}{\sigma_{\text{downside}}}$$

Where:
- **MAR (Minimum Acceptable Return)**: The threshold below which returns are considered "bad." Often set to 0 (any loss is bad) or the risk-free rate. MAR is a user-defined benchmark — it answers the question "what return do I need to be satisfied?"
- **Downside Deviation** ($\sigma_{\text{downside}}$): Standard deviation computed only using returns below the MAR.

$$\sigma_{\text{downside}} = \sqrt{\frac{1}{T} \sum_{r_t < \text{MAR}} (r_t - \text{MAR})^2}$$

**Why Sortino > Sharpe for evaluation?** Consider two portfolios:
- Portfolio A: returns = [+5%, +3%, -1%, +4%, -2%] → σ = 2.86%
- Portfolio B: returns = [+1%, +1%, +1%, +1%, +1%] → σ = 0%

Portfolio A has higher volatility but it's mostly upside. Sharpe penalizes this. Sortino correctly recognizes that A's "risk" is mostly positive.

**In our project**: We use MAR = 0 (the risk-free rate), so any negative return counts as downside risk.

---

## 7. Downside Risk Metrics

These metrics focus specifically on losses, which is what investors actually care about.

### Maximum Drawdown (MDD)

The largest peak-to-trough decline in portfolio value.

$$\text{MDD} = \min_t \left(\frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}\right)$$

**In plain English**: Find the worst "high point to low point" drop across the entire period.

**Example**:
```
Portfolio value: $100 → $120 → $90 → $110 → $80 → $130
                              ↑ peak at $120
                                       ↓ trough at $80
MDD = (80 - 120) / 120 = -33.3%
```

**Why it matters**: Even if your total return is great, a -50% drawdown means an investor who joined at the peak lost half their money. Many investors can't stomach that and will sell at the worst time.

| MDD | Interpretation |
|-----|----------------|
| > -10% | Excellent risk control |
| -10% to -20% | Normal for diversified portfolios |
| -20% to -40% | Significant, typical in recessions |
| < -40% | Severe, potential recovery takes years |

### Value-at-Risk (VaR)

**"What's the worst daily loss I should expect under normal conditions?"**

$$\text{VaR}_\alpha = \text{quantile}(r, 1 - \alpha)$$

For 95% VaR: the 5th percentile of the return distribution.

**Example**: VaR₉₅ = -1.74% means "on 95% of days, you'll lose no more than 1.74%. On the worst 5% of days, losses exceed this."

**Limitation**: VaR tells you the threshold, but nothing about HOW BAD it gets beyond that threshold.

### Conditional VaR (CVaR) / Expected Shortfall

**"When things go bad (beyond VaR), how bad do they get on average?"**

$$\text{CVaR}_\alpha = \mathbb{E}[r \mid r \leq \text{VaR}_\alpha]$$

The average loss in the worst (1-α)% of days.

**Example**: CVaR₉₅ = -2.78% means "on the worst 5% of days, the average loss is 2.78%."

CVaR is always more extreme than VaR (by definition). It's considered a better risk measure because:
1. It captures tail risk (how bad the worst days are)
2. It's a **coherent risk measure** (satisfies mathematical properties that VaR doesn't)

### Portfolio Turnover

$$\text{Turnover} = \frac{1}{T} \sum_t \sum_i |w_{i,t} - w_{i,t-1}|$$

Average total change in weights per rebalance. High turnover = high transaction costs.

**Example**: If you completely flip your portfolio (sell everything, buy different things), turnover = 2.0 (200%). Typical acceptable turnover for monthly rebalancing is 0.05-0.20.

---

## 8. Volatility Regimes

Financial markets don't behave the same way all the time. They alternate between **regimes**:

### Low Volatility Regime
- Small daily moves (< 1%)
- Calm, trending markets
- Typical in bull markets (2013-2014, 2017, 2021)
- Correlations between assets are lower
- MVO tends to work well here

### Medium Volatility Regime
- Normal market conditions
- Moderate daily moves (1-2%)
- Most of the time markets are here

### High Volatility Regime
- Large daily moves (2-5%+, sometimes 10%+)
- Market crises, panics
- Examples: 2008 Financial Crisis, March 2020 (COVID), 2022 rate hikes
- **Correlations spike** — everything crashes together
- MVO performs worst here (assumptions break down)
- This is where LSTM might add value by detecting regime shifts

### How We Identify Regimes

**Method 1: VIX-Based** (our primary method)

The **VIX** (CBOE Volatility Index) is called the "fear gauge." It measures the market's expectation of 30-day volatility, derived from S&P 500 option prices.

| VIX Level | Regime | Market Mood |
|-----------|--------|-------------|
| < 15 | Low | Calm, complacent |
| 15 – 25 | Medium | Normal uncertainty |
| > 25 | High | Fear, stress |
| > 40 | Extreme | Panic (rare) |

VIX hit **82.69** during the COVID crash (March 16, 2020) — the highest ever recorded.

**Method 2: Rolling Standard Deviation**

Compute the rolling 252-day (1 year) standard deviation of returns, then classify into terciles (bottom 33% = low, top 33% = high).

This doesn't rely on VIX data and works for any asset, but it's backward-looking (lags regime changes).

### Why Regimes Matter for This Project

Our central research question is whether LSTM can outperform MVO **especially during high-volatility regimes**, where:
- MVO's assumption of stable parameters breaks down
- Nonlinear dependencies emerge
- LSTM might capture temporal patterns that linear models miss

---

## 9. Backtesting

**Backtesting** = simulating how a strategy would have performed on historical data.

### Walk-Forward (Rolling Window) Backtesting

Our approach:

```
Time ──────────────────────────────────────────────────►

[=== Training Window (252 days) ===][OOS]
                                    ↑ Optimize weights here
                                    ↑ Evaluate return here (out-of-sample)

    [=== Training Window (252 days) ===][OOS]
                                        ↑ Re-optimize
                                        ↑ Evaluate

        [=== Training Window (252 days) ===][OOS]
            ... and so on, rebalancing every 21 days (monthly)
```

**Key features**:
1. **Rolling window**: Always train on the most recent 252 days. This lets the model adapt to changing market conditions.
2. **Out-of-sample evaluation**: The return we measure is ALWAYS on data the model hasn't seen. This prevents overfitting.
3. **Rebalancing frequency**: We re-optimize weights every 21 trading days (~monthly). Between rebalances, weights drift naturally with market moves.

### Why Not Just Train Once and Test?

Markets change (non-stationary). A covariance matrix from 2010 is useless in 2020. Rolling windows let the model continuously adapt.

### Pitfalls of Backtesting

1. **Look-ahead bias**: Using future information in past decisions. Our walk-forward design prevents this.
2. **Survivorship bias**: Only testing on assets that still exist today (the ones that didn't go bankrupt). Sector ETFs mitigate this since they automatically rebalance.
3. **Overfitting**: Tweaking strategy parameters until it works on historical data. Bailey et al. (2014) showed this is rampant. Our test set is held out to detect this.
4. **Transaction costs**: Ignoring the cost of trading. We measure turnover to estimate this impact.

---

## 10. LSTM and Deep Learning for Finance

### Why Deep Learning for Portfolio Optimization?

MVO assumes returns are linearly related and come from a stable distribution. In reality:
- Returns have **nonlinear dependencies** (e.g., momentum effects, mean reversion)
- Markets have **temporal patterns** (trends, cycles, regime persistence)
- Relationships between assets **change over time**

**LSTM (Long Short-Term Memory)** networks can potentially capture these patterns.

### What Is an LSTM?

An LSTM is a type of **Recurrent Neural Network (RNN)** designed to learn long-term dependencies in sequential data.

**Standard RNN problem**: When trying to learn from long sequences, gradients either vanish (→ 0) or explode (→ ∞) during backpropagation through time. The network can't learn long-range patterns.

**LSTM solution**: Add a **cell state** — a highway that carries information across many time steps with minimal transformation. Three gates control what information flows in, out, and what's forgotten:

```
         ┌─────────────────────────────────┐
         │           Cell State (Cₜ)        │ ← long-term memory highway
         │  ┌───────┐  ┌──────┐  ┌──────┐  │
Input ──►│  │Forget │  │Input │  │Output│  │──► Hidden State (hₜ)
(xₜ)     │  │ Gate  │  │ Gate │  │ Gate │  │    (short-term output)
         │  │  fₜ   │  │  iₜ  │  │  oₜ  │  │
         │  └───────┘  └──────┘  └──────┘  │
         └─────────────────────────────────┘
```

**Forget Gate** ($f_t$): Decides what to discard from cell state.
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** ($i_t$): Decides what new information to store.
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell State Update**:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output Gate** ($o_t$): Decides what to output.
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

Where $\sigma$ = sigmoid function, $\odot$ = element-wise multiplication.

### How LSTM Is Used in Our Project

1. **Input**: Rolling window of past N days of returns for all assets (sequence of N×11 vectors)
2. **Output**: Predicted next-day returns for each asset (11 values)
3. **Portfolio construction**: Use predicted returns either:
   - **Directly**: Allocate proportionally to predicted risk-adjusted returns
   - **As MVO input**: Replace historical mean returns with LSTM predictions in the optimization

### Key Hyperparameters

| Parameter | Typical Range | What It Controls |
|-----------|---------------|------------------|
| Sequence length | 20-60 days | How far back the LSTM looks |
| Hidden units | 32-128 | Model capacity (too many → overfit) |
| Dropout rate | 0.1-0.3 | Regularization (randomly zero out neurons) |
| Learning rate | 1e-4 to 1e-2 | Step size during training |
| Batch size | 32-128 | Samples per gradient update |

### Why LSTM Might Fail

It's important to be honest about challenges:
1. **Financial markets are noisy**: Signal-to-noise ratio is very low compared to NLP or computer vision
2. **Non-stationarity**: Patterns that worked in the past may not persist
3. **Overfitting**: With limited financial data, deep models easily memorize rather than generalize
4. **Transaction costs**: Even if LSTM improves predictions slightly, frequent trading could eat the gains
5. **The Efficient Market Hypothesis**: If patterns were easily exploitable, they'd be arbitraged away

---

## 11. Glossary

| Term | Definition |
|------|-----------|
| **Alpha** | Excess return above a benchmark (e.g., S&P 500) |
| **Asset** | Anything that can be owned and has value (stocks, bonds, real estate, etc.) |
| **Backtest** | Simulating a strategy on historical data to evaluate performance |
| **Benchmark** | A reference portfolio to compare against (often S&P 500 or equal-weight) |
| **Bull/Bear Market** | Sustained period of rising/falling prices |
| **Correlation** | Normalized measure of how two assets move together (-1 to +1) |
| **Covariance Matrix** | Matrix of pairwise covariances between all assets |
| **CVaR** | Conditional Value-at-Risk; average loss in the worst (1-α)% of days |
| **Diversification** | Reducing risk by holding multiple assets that don't move identically |
| **Drawdown** | Peak-to-trough decline in portfolio value |
| **Efficient Frontier** | Set of portfolios with maximum return for each risk level |
| **ETF** | Exchange-Traded Fund; a basket of assets traded as one security |
| **GICS** | Global Industry Classification Standard; the sector classification system |
| **Log Return** | ln(P_t / P_{t-1}); additive over time |
| **Long-Only** | Portfolio that only buys assets (no short selling); weights ≥ 0 |
| **MAR** | Minimum Acceptable Return; the threshold for Sortino ratio (often 0 or risk-free rate) |
| **MDD** | Maximum Drawdown; largest peak-to-trough portfolio decline |
| **MVO** | Mean-Variance Optimization; Markowitz's framework |
| **Out-of-Sample** | Data not used during model training; used for honest evaluation |
| **Rebalancing** | Adjusting portfolio weights back to target allocations |
| **Risk-Free Rate** | Return on a "riskless" investment (e.g., US Treasury bills) |
| **Rolling Window** | A fixed-size window that slides forward through time |
| **Sharpe Ratio** | (Return - Risk-Free Rate) / Volatility; return per unit of risk |
| **Short Selling** | Selling an asset you don't own (borrowing it, hoping to buy back cheaper) |
| **Sortino Ratio** | Like Sharpe but using only downside deviation |
| **Turnover** | How much portfolio weights change at each rebalance |
| **VaR** | Value-at-Risk; worst expected loss at a confidence level |
| **VIX** | CBOE Volatility Index; market's expected 30-day volatility |
| **Volatility** | Standard deviation of returns; primary measure of risk |
| **Walk-Forward** | Backtesting method that slides the training window forward in time |

---

## References

- Markowitz, H. (1952). *Portfolio Selection.* The Journal of Finance, 7(1), 77–91.
- Sharpe, W. F. (1994). *The Sharpe Ratio.* Journal of Portfolio Management, 21(1), 49–58.
- Sortino, F. A., & Van der Meer, R. (1991). *Downside Risk.* Journal of Portfolio Management, 17(4), 27–31.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- Fischer, T., & Krauss, C. (2018). *Deep Learning with LSTM Networks for Financial Market Predictions.* European Journal of Operational Research, 270(2), 654–669.
- Whaley, R. E. (2000). *The Investor Fear Gauge.* Journal of Portfolio Management, 26(3), 12–17.
- Bailey, D. H., et al. (2014). *The Probability of Backtest Overfitting.* Journal of Computational Finance, 20(4), 39–69.
