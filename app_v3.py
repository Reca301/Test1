import io
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Financial Analytics Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- NAVIGATION SELECTOR (Sidebar) ---
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio(
    "Choose Tool:",
    ["AGX1 Factor Attribution", "SAA Portfolio Optimizer"]
)

st.sidebar.markdown("---")

# =============================================================================
# TOOL 1: AGX1 FACTOR ATTRIBUTION
# =============================================================================
if app_mode == "AGX1 Factor Attribution":
    st.title("📊 AGX1 4 Factor Attribution")
    st.markdown("This tool reads data from `Test1/AGX1_test.xlsx`, runs expanding-window OLS regressions, and displays split attribution charts and tables.")

    FILE_PATH = "data/AGX1_test.xlsx"

    @st.cache_data
    def run_attribution(path):
        required_cols = ["EOM", "AGX1_MSCI", "MSCI_RBA", "SMB", "HML", "UMD"]
        
        df_raw = (
            pd.read_excel(path, sheet_name="Data")[required_cols]
            .dropna()
            .sort_values("EOM")
            .reset_index(drop=True)
        )
        df_raw["EOM"] = pd.to_datetime(df_raw["EOM"])

        attribution_results = []

        for i in range(35, len(df_raw)):
            df_hist = df_raw.iloc[: i + 1].copy()
            y = df_hist["AGX1_MSCI"]
            X = sm.add_constant(df_hist[["MSCI_RBA", "SMB", "HML", "UMD"]])
            model = sm.OLS(y, X).fit()

            alpha_active = model.params["const"]
            beta_active = model.params["MSCI_RBA"]
            size_p = model.params["SMB"]
            value_p = model.params["HML"]
            momentum_p = model.params["UMD"]

            current_row = df_raw.iloc[i]
            agx1_return = current_row["AGX1_MSCI"]
            msci_rba = current_row["MSCI_RBA"]
            smb = current_row["SMB"]
            hml = current_row["HML"]
            umd = current_row["UMD"]

            benchmark_contribution = beta_active * msci_rba
            size_contribution = size_p * smb
            value_contribution = value_p * hml
            momentum_contribution = momentum_p * umd

            residual = (
                agx1_return
                - alpha_active
                - benchmark_contribution
                - size_contribution
                - value_contribution
                - momentum_contribution
            )

            attribution_results.append({
                "EOM": current_row["EOM"],
                "AGX1_MSCI": agx1_return,
                "Alpha_Active": alpha_active,
                "Coeff_Beta_Active": beta_active,
                "Coeff_Size": size_p,
                "Coeff_Value": value_p,
                "Coeff_Momentum": momentum_p,
                "P_Alpha": model.pvalues["const"],
                "P_Beta_Active": model.pvalues["MSCI_RBA"],
                "P_Size": model.pvalues["SMB"],
                "P_Value": model.pvalues["HML"],
                "P_Momentum": model.pvalues["UMD"],
                "R_Squared": model.rsquared,
                "Adj_R_Squared": model.rsquared_adj,
                "Benchmark": benchmark_contribution,
                "Size": size_contribution,
                "Value": value_contribution,
                "Momentum": momentum_contribution,
                "Res": residual,
            })

        df_attr = pd.DataFrame(attribution_results)
        numeric_cols = df_attr.select_dtypes(include="number").columns
        df_attr[numeric_cols] = df_attr[numeric_cols].round(6)

        df_attr["Reconstructed_Return"] = (
            df_attr["Alpha_Active"]
            + df_attr["Benchmark"]
            + df_attr["Size"]
            + df_attr["Value"]
            + df_attr["Momentum"]
            + df_attr["Res"]
        )
        return df_attr

    if st.sidebar.button("Run Factor Attribution", type="primary"):
        with st.spinner("Processing expanding-window regressions..."):
            df_attribution = run_attribution(FILE_PATH)

        st.success("Attribution calculations completed successfully!")

        # Filter for the most recent 12 periods
        df_chart = df_attribution.tail(12).copy()
        df_chart["EOM_Str"] = df_chart["EOM"].dt.strftime("%Y-%m")

        # --- CHART 1: Stacked Bar Chart of Return Contributions ---
        st.subheader("📈 Factor Return Contributions (Last 12 Periods)")
        
        fig_bar = go.Figure()
        components = [
            ("Alpha", "Alpha_Active", "#2ecc71"),
            ("Benchmark", "Benchmark", "#3498db"),
            ("Size (SMB)", "Size", "#9b59b6"),
            ("Value (HML)", "Value", "#e67e22"),
            ("Momentum (UMD)", "Momentum", "#e74c3c"),
            ("Residual", "Res", "#95a5a6")
        ]

        for name, col, color in components:
            fig_bar.add_trace(
                go.Bar(
                    x=df_chart["EOM_Str"],
                    y=df_chart[col],
                    name=name,
                    marker_color=color
                )
            )

        fig_bar.update_layout(
            barmode="relative",
            xaxis_title="End of Month (EOM)",
            yaxis_title="Return Contribution",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- CHART 2: Adjusted R-Squared Line Chart (Matrix Green & Bold) ---
        st.subheader("📉 Adjusted R-Squared (Last 12 Periods)")
        
        fig_r2 = go.Figure()
        fig_r2.add_trace(
            go.Scatter(
                x=df_chart["EOM_Str"],
                y=df_chart["Adj_R_Squared"],
                name="Adj R-Squared",
                mode="lines+markers",
                line=dict(color="#00FF41", width=4),  # Matrix green in bold
                marker=dict(size=8, color="#00FF41")
            )
        )

        fig_r2.update_layout(
            xaxis_title="End of Month (EOM)",
            yaxis_title="Adjusted R-Squared",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        # --- DATA TABLES ---
        st.subheader("📋 Attribution Results Table")
        st.dataframe(df_attribution, use_container_width=True)

        st.subheader("🔍 Reconciliation Check (Last 10 Observations)")
        st.dataframe(
            df_attribution[["EOM", "AGX1_MSCI", "Reconstructed_Return"]].tail(10),
            use_container_width=True,
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_attribution.to_excel(writer, sheet_name="Attribution", index=False)

        st.download_button(
            label="📥 Download Attribution Results as Excel",
            data=buffer.getvalue(),
            file_name="AGX1_Attribution_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("👈 Click **Run Factor Attribution** in the sidebar to load your dataset from `Test1/AGX1_test.xlsx`.")

# =============================================================================
# TOOL 2: SAA PORTFOLIO OPTIMIZER
# =============================================================================
elif app_mode == "SAA Portfolio Optimizer":
    st.title("📈 ASX Vanguard Strategic Asset Allocation (SAA) Optimizer")
    st.markdown("Select your preferred Vanguard ASX ETFs, set your risk-free rate, and calculate the efficient frontier using interactive Plotly charts.")

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
        format="%.2f"
    )
    risk_free_rate = rf_percentage / 100.0

    optimize_button = st.sidebar.button("Optimize SAA", type="primary")

    if optimize_button:
        if len(selected_display_names) < 2:
            st.error("⚠️ Please select at least 2 ETFs in the sidebar to run the optimization.")
        else:
            selected_etfs = [etf_options[name] for name in selected_display_names]
            
            with st.spinner("Fetching historical data from Yahoo Finance and simulating 5,000 portfolios..."):
                data = yf.download(selected_etfs, period="15y", auto_adjust=False, progress=False)
                
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
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=portfolios_df['Volatility'], y=portfolios_df['Returns'],
                        mode='markers', marker=dict(size=6, color='#3498db', opacity=0.5),
                        name='Simulated Portfolios'
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_efficient_front['Volatility'], y=df_efficient_front['Returns'],
                        mode='markers', marker=dict(size=8, color='#2ecc71'),
                        name='Efficient Frontier'
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines', line=dict(color='#e74c3c', width=2, dash='dash'),
                        name='Capital Market Line (CML)'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[optimal_volatility], y=[optimal_return],
                        mode='markers', marker=dict(size=16, color='#e74c3c', symbol='star'),
                        name='Optimal Portfolio'
                    ))
                    fig.update_layout(
                        title='<b>Efficient Frontier & Capital Market Line (CML)</b>',
                        xaxis_title='Annualized Volatility (Standard Deviation)',
                        yaxis_title='Annualized Return',
                        template='plotly_dark',
                        hovermode='closest'
                    )
                    
                    st.success("Optimization completed successfully!")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 🎯 Optimal Portfolio Metrics & Allocation")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Annual Return", f"{optimal_return*100:.2f}%")
                    col2.metric("Annualized Volatility", f"{optimal_volatility*100:.2f}%")
                    col3.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
                    
                    # Calculate individual ETF data periods, average returns, and standard deviations
                    etf_table_data = []
                    for t in selected_etfs:
                        # Extract ticker prices series
                        if isinstance(prices, pd.DataFrame):
                            p_series = prices[t].dropna()
                        else:
                            p_series = prices.dropna()
                            
                        start_date = p_series.index.min().strftime('%Y-%m-%d')
                        end_date = p_series.index.max().strftime('%Y-%m-%d')
                        period_str = f"{start_date} to {end_date}"
                        
                        # Annualized return and std dev for this specific ticker
                        t_returns = daily_returns[t] if isinstance(daily_returns, pd.DataFrame) else daily_returns
                        ann_ret = t_returns.mean() * 252
                        ann_std = t_returns.std() * np.sqrt(252)
                        
                        # Find ticker display name
                        display_name = [k for k, v in etf_options.items() if v == t][0]
                        ticker_code = display_name.split(" ")[0]
                        
                        etf_table_data.append({
                            "ETF Ticker": ticker_code,
                            "Data Period": period_str,
                            "Average Annual Return (%)": round(ann_ret * 100, 2),
                            "Annualized Standard Deviation (%)": round(ann_std * 100, 2),
                            "Optimal Weight (%)": round(optimal_portfolio[f'{t}_Wt'] * 100, 2)
                        })
                    
                    weights_df = pd.DataFrame(etf_table_data)
                    st.dataframe(weights_df, use_container_width=True)
    else:
        st.info("👈 Select your ETFs and Risk-Free Rate in the sidebar, then click **Optimize SAA** to begin.")
