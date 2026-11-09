import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="ASX Vanguard SAA Optimizer",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ASX Vanguard Strategic Asset Allocation (SAA) Optimizer")
st.markdown("Select your preferred Vanguard ASX ETFs, set your risk-free rate, and calculate the efficient frontier using interactive Plotly charts and historical market data.")

# --- SIDEBAR & INPUTS ---
st.sidebar.header("Portfolio Parameters")

etf_options = {
    "VAS (Australian Shares)": "VAS.AX",
    "VHY (High Yield Australian Shares)": "VHY.AX",
    "VGS (International Shares)": "VGS.AX",
    "VGAD (International Shares - Hedged)": "VGAD.AX",
    "V500 (S&P 500 ETF)": "V500.AX",
    "V5AH (S&P 500 ETF - Hedged)": "V5AH.AX",
    "VAF (Australian Fixed Interest)": "VAF.AX",
    "VAP (Australian Property Securities)": "VAP.AX",
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

optimize_button = st.sidebar.button("Optimize SAA", type="primary")

# --- MAIN EXECUTION ---
if optimize_button:
    if len(selected_display_names) < 2:
        st.error("⚠️ Please select at least 2 ETFs in the sidebar to run the optimization.")
    else:
        selected_etfs = [etf_options[name] for name in selected_display_names]
        
        with st.spinner("Fetching historical data from Yahoo Finance and simulating 5,000 portfolios..."):
            data = yf.download(selected_etfs, period="5y", auto_adjust=False, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Adj Close']
            else:
                prices = data[['Adj Close']] if 'Adj Close' in data.columns else data
                
            prices = prices.dropna()
            
            if prices.empty or len(prices) < 30:
                st.error("Could not fetch sufficient historical price data. Please verify your ticker selections.")
            else:
                daily_returns = prices.pct_change().dropna()
                annualized_returns = daily_returns.mean() * 252
                covariance_matrix = daily_returns.cov() * 252
                
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
                
                portfolios_sorted = portfolios_df.sort_values(by='Volatility', ascending=True)
                max_returns_so_far = -np.inf
                efficient_portfolios_list = []
                
                for _, row in portfolios_sorted.iterrows():
                    if row['Returns'] > max_returns_so_far:
                        efficient_portfolios_list.append(row)
                        max_returns_so_far = row['Returns']
                        
                df_efficient_front = pd.DataFrame(efficient_portfolios_list)
                
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
                
                # --- Create Interactive Plotly Chart ---
                fig = go.Figure()

                # 1. Simulated Portfolios
                fig.add_trace(go.Scatter(
                    x=portfolios_df['Volatility'],
                    y=portfolios_df['Returns'],
                    mode='markers',
                    marker=dict(size=6, color='#3498db', opacity=0.5),
                    name='Simulated Portfolios',
                    hovertemplate='Volatility: %{x:.2f}<br>Return: %{y:.2f}<extra></extra>'
                ))

                # 2. Efficient Frontier Portfolios
                fig.add_trace(go.Scatter(
                    x=df_efficient_front['Volatility'],
                    y=df_efficient_front['Returns'],
                    mode='markers',
                    marker=dict(size=8, color='#2ecc71'),
                    name='Efficient Frontier',
                    hovertemplate='Volatility: %{x:.2f}<br>Return: %{y:.2f}<extra></extra>'
                ))

                # 3. Capital Market Line (CML)
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    line=dict(color='#e74c3c', width=2, dash='dash'),
                    name='Capital Market Line (CML)'
                ))

                # 4. Optimal Portfolio Marker
                fig.add_trace(go.Scatter(
                    x=[optimal_volatility],
                    y=[optimal_return],
                    mode='markers',
                    marker=dict(size=16, color='#e74c3c', symbol='star'),
                    name='Optimal Portfolio',
                    hovertemplate='<b>Optimal Portfolio</b><br>Volatility: %{x:.2f}<br>Return: %{y:.2f}<extra></extra>'
                ))

                # Layout Styling
                fig.update_layout(
                    title='<b>Efficient Frontier & Capital Market Line (CML)</b>',
                    xaxis_title='Annualized Volatility (Standard Deviation)',
                    yaxis_title='Annualized Return',
                    template='plotly_white',
                    hovermode='closest',
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
                )
                
                st.success("Optimization completed successfully!")
                st.plotly_chart(fig, use_container_width=True)
                
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
