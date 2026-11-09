import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="ASX Vanguard SAA Optimizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ASX Vanguard Strategic Asset Allocation (SAA) Optimizer")
st.markdown("Select your preferred Vanguard ASX ETFs, set your risk-free rate, and calculate the efficient frontier using historical market data.")

# --- SIDEBAR & INPUTS ---
st.sidebar.header("Portfolio Parameters")

# 1a. Input selection - 10 Vanguard ETFs listed on ASX
etf_options = {
    "VAS (Australian Shares)": "VAS.AX",
    "VHY (High Yield Australian Shares)": "VHY.AX",
    "VGS (International Shares)": "VGS.AX",
    "VGAD (International Shares - Hedged)": "VGAD.AX",
    "V500 (S&P 500 ETF)": "V500.AX",
    "V5AH (S&P 500 ETF - Hedged)": "V5AH.AX",
    "VAF (Australian Fixed Interest)": "VAF.AX",
    "VFLT (Diversified Fixed Interest)": "VFLT.AX",
    "VTS (US Total Market Shares)": "VTS.AX",
    "VEU (All World ex-US Shares)": "VEU.AX"
}

selected_display_names = st.sidebar.multiselect(
    "Select ETFs (Choose 2 to 10):",
    options=list(etf_options.keys()),
    default=[
        "VAS (Australian Shares)",
        "VGS (International Shares)",
        "VAF (Australian Fixed Interest)"
    ]
)

# 1b2. risk_free_rate input (percentage with 2 decimal places)
rf_percentage = st.sidebar.number_input(
    "Risk-Free Rate (%):",
    min_value=0.00,
    max_value=20.00,
    value=4.25,
    step=0.01,
    format="%.2f",
    help="Enter percentage number with 2 decimal places (e.g., 4.25 for 4.25%)"
)
risk_free_rate = rf_percentage / 100.0

# 1c. Optimize SAA Button
optimize_button = st.sidebar.button("Optimize SAA", type="primary")

# --- MAIN EXECUTION ---
if optimize_button:
    if len(selected_display_names) < 2:
        st.error("⚠️ Please select at least 2 ETFs in the sidebar to run the optimization.")
    else:
        selected_etfs = [etf_options[name] for name in selected_display_names]
        
        with st.spinner("Fetching historical data from Yahoo Finance and simulating 5,000 portfolios..."):
            # 2a. Pull historical price data (using Adj Close)
            data = yf.download(selected_etfs, period="5y", auto_adjust=False, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                prices = data[['Adj Close']] if 'Adj Close' in data.columns else data
                
            prices = prices.dropna()
            
            if prices.empty or len(prices) < 30:
                st.error("Could not fetch sufficient historical price data. Please verify your ticker selections.")
            else:
                # 2b. Calculate daily returns
                daily_returns = prices.pct_change().dropna()
                
                # 2c. Calculate historical annual returns and annual standard deviation
                annualized_returns = daily_returns.mean() * 252
                
                # 2d. Calculate pair-wise covariance matrix
                covariance_matrix = daily_returns.cov() * 252
                
                # 2e. Generate 5000 random portfolios
                num_portfolios = 5000
                portfolio_metrics = {
                    'Returns': [],
                    'Volatility': [],
                    **{f'{name}_Wt': [] for name in selected_etfs}
                }
                
                np.random.seed(42)
                for _ in range(num_portfolios):
                    weights = np.random.random(len(selected_etfs))
                    weights /= np.sum(weights)
                    
                    portfolio_return = np.dot(weights, annualized_returns)
                    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
                    
                    portfolio_metrics['Returns'].append(portfolio_return)
                    portfolio_metrics['Volatility'].append(portfolio_std_dev)
                    
                    for i, name in enumerate(selected_etfs):
                        portfolio_metrics[f'{name}_Wt'].append(weights[i])
                
                portfolios_df = pd.DataFrame(portfolio_metrics)
                
                # 2e3. Identify efficient frontier portfolios
                portfolios_sorted = portfolios_df.sort_values(by='Volatility', ascending=True)
                max_returns_so_far = -np.inf
                efficient_portfolios_list = []
                
                for _, row in portfolios_sorted.iterrows():
                    if row['Returns'] > max_returns_so_far:
                        efficient_portfolios_list.append(row)
                        max_returns_so_far = row['Returns']
                        
                df_efficient_front = pd.DataFrame(efficient_portfolios_list)
                
                # 2e4. Add Capital Market Line and filter outliers
                df_efficient_front['Sharpe_Ratio'] = (df_efficient_front['Returns'] - risk_free_rate) / df_efficient_front['Volatility']
                
                Q1 = df_efficient_front['Sharpe_Ratio'].quantile(0.25)
                Q3 = df_efficient_front['Sharpe_Ratio'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.25 * IQR
                upper_bound = Q3 + 1.25 * IQR
                vol_cap = 0.9 * df_efficient_front['Volatility'].max()
                
                df_ef_filtered = df_efficient_front[
                    (df_efficient_front['Sharpe_Ratio'] >= lower_bound) &
                    (df_efficient_front['Sharpe_Ratio'] <= upper_bound) &
                    (df_efficient_front['Volatility'] <= vol_cap)
                ]
                
                if df_ef_filtered.empty:
                    df_ef_filtered = df_efficient_front
                    
                optimal_portfolio = df_ef_filtered.loc[df_ef_filtered['Sharpe_Ratio'].idxmax()]
                optimal_return = optimal_portfolio['Returns']
                optimal_volatility = optimal_portfolio['Volatility']
                optimal_sharpe = optimal_portfolio['Sharpe_Ratio']
                
                x_line = np.linspace(0, max(portfolios_df['Volatility']) + 0.01, 100)
                y_line = optimal_sharpe * x_line + risk_free_rate
                
                # 2e5. Modern Chart style on output panel
                plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                
                ax.scatter(
                    portfolios_df['Volatility'],
                    portfolios_df['Returns'],
                    c='lightblue',
                    alpha=0.6,
                    s=18,
                    label='Simulated Portfolios'
                )
                ax.scatter(
                    df_efficient_front['Volatility'],
                    df_efficient_front['Returns'],
                    c='#2ecc71',
                    s=28,
                    label='Efficient Frontier'
                )
                ax.plot(
                    x_line,
                    y_line,
                    c='#e74c3c',
                    linestyle='--',
                    linewidth=2,
                    label='Capital Market Line (CML)'
                )
                ax.scatter(
                    optimal_volatility,
                    optimal_return,
                    c='#e74c3c',
                    marker='*',
                    s=250,
                    zorder=5,
                    label='Optimal Portfolio'
                )
                
                ax.set_title('Efficient Frontier & Capital Market Line (CML)', fontsize=14, fontweight='bold', pad=15)
                ax.set_xlabel('Annualized Volatility (Standard Deviation)', fontsize=11, fontweight='semibold')
                ax.set_ylabel('Annualized Return', fontsize=11, fontweight='semibold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=10)
                plt.tight_layout()
                
                st.success("Optimization completed successfully!")
                st.pyplot(fig)
                
                # Results breakdown
                st.markdown("### 🎯 Optimal Portfolio Metrics & Allocation")
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Annual Return", f"{optimal_return*100:.2f}%")
                col2.metric("Annualized Volatility", f"{optimal_volatility*100:.2f}%")
                col3.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
                
                # Weights table
                weights_df = pd.DataFrame({
                    "ETF Ticker": [name.split(" ")[0] for name in selected_display_names],
                    "Optimal Weight (%)": [round(optimal_portfolio[f'{t}_Wt'] * 100, 2) for t in selected_etfs]
                })
                st.dataframe(weights_df, use_container_width=True)
else:
    st.info("👈 Select your ETFs and Risk-Free Rate in the sidebar, then click **Optimize SAA** to begin.")
