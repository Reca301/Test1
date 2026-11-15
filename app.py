import io
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

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
    [
        "AGX1 Factor Attribution", 
        "SAA Portfolio Optimizer", 
        "Fixed Income Attribution"
    ]
)

st.sidebar.markdown("---")

# =============================================================================
# TOOL 1: AGX1 FACTOR ATTRIBUTION (OLS & KALMAN FILTER)
# =============================================================================
if app_mode == "AGX1 Factor Attribution":
    st.title("📊 AGX1 Factor Attribution (OLS vs. Kalman Filter)")
    st.markdown("This tool reads data from `data/AGX1_test.xlsx` and performs factor attribution using either Expanding-Window OLS or a Time-Varying Kalman Filter model.")

    model_choice = st.sidebar.selectbox(
        "Attribution Model:",
        ["Expanding-Window OLS", "Time-Varying Kalman Filter"]
    )

    FILE_PATH = "data/AGX1_test.xlsx"

    @st.cache_data
    def run_attribution(path, model_type):
        required_cols = ["EOM", "AGX1_MSCI", "MSCI_RBA", "SMB", "HML", "UMD"]
        
        df_raw = (
            pd.read_excel(path, sheet_name="Data")[required_cols]
            .dropna()
            .sort_values("EOM")
            .reset_index(drop=True)
        )
        df_raw["EOM"] = pd.to_datetime(df_raw["EOM"])

        attribution_results = []

        if model_type == "Expanding-Window OLS":
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

        else:
            y = df_raw["AGX1_MSCI"].values
            X = sm.add_constant(df_raw[["MSCI_RBA", "SMB", "HML", "UMD"]]).values
            
            n_obs, n_features = X.shape
            theta = np.zeros(n_features)
            P = np.eye(n_features) * 1.0
            R = 1e-3
            Q = np.eye(n_features) * 1e-5
            
            recent_errors = []
            
            for i in range(n_obs):
                x_t = X[i]
                y_t = y[i]
                current_row = df_raw.iloc[i]
                
                if i >= 35:
                    y_pred = np.dot(x_t, theta)
                    error = y_t - y_pred
                    recent_errors.append(error**2)
                    if len(recent_errors) > 12:
                        recent_errors.pop(0)
                    
                    sigma2 = np.mean(recent_errors) if recent_errors else R
                    
                    P_x = np.dot(P, x_t)
                    s = np.dot(x_t, P_x) + R
                    K = P_x / s
                    
                    theta = theta + K * error
                    P = (np.eye(n_features) - np.outer(K, x_t)) @ P + Q
                    
                    alpha_active = theta[0]
                    beta_active = theta[1]
                    size_p = theta[2]
                    value_p = theta[3]
                    momentum_p = theta[4]
                    
                    param_ses = np.sqrt(np.maximum(1e-8, np.diag(P) * sigma2))
                    t_stats = theta / param_ses
                    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
                    
                    benchmark_contribution = beta_active * current_row["MSCI_RBA"]
                    size_contribution = size_p * current_row["SMB"]
                    value_contribution = value_p * current_row["HML"]
                    momentum_contribution = momentum_p * current_row["UMD"]
                    
                    y_hist_slice = y[max(0, i-35):i+1]
                    err_hist_slice = np.array(recent_errors) if recent_errors else np.array([error**2])
                    r_sq = max(0.0, min(0.99, 1.0 - (np.mean(err_hist_slice) / max(1e-6, np.var(y_hist_slice)))))
                    adj_r_sq = max(0.0, r_sq - (n_features / len(y_hist_slice)))
                    
                    attribution_results.append({
                        "EOM": current_row["EOM"],
                        "AGX1_MSCI": y_t,
                        "Alpha_Active": alpha_active,
                        "Coeff_Beta_Active": beta_active,
                        "Coeff_Size": size_p,
                        "Coeff_Value": value_p,
                        "Coeff_Momentum": momentum_p,
                        "P_Alpha": p_values[0],
                        "P_Beta_Active": p_values[1],
                        "P_Size": p_values[2],
                        "P_Value": p_values[3],
                        "P_Momentum": p_values[4],
                        "R_Squared": r_sq,
                        "Adj_R_Squared": adj_r_sq,
                        "Benchmark": benchmark_contribution,
                        "Size": size_contribution,
                        "Value": value_contribution,
                        "Momentum": momentum_contribution,
                        "Res": error,
                    })
                else:
                    theta += 0.01 * x_t * (y_t - np.dot(x_t, theta))

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

    if st.sidebar.button("Run Attribution Model", type="primary"):
        with st.spinner(f"Running {model_choice}..."):
            df_attribution = run_attribution(FILE_PATH, model_choice)

        st.success(f"{model_choice} calculations completed successfully!")

        df_chart = df_attribution.tail(12).copy()
        df_chart["EOM_Str"] = df_chart["EOM"].dt.strftime("%Y-%m")

        st.subheader(f"📈 Factor Return Contributions ({model_choice} - Last 12 Periods)")
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

        st.subheader("📉 Fit Metric (Last 12 Periods)")
        fig_r2 = go.Figure()
        fig_r2.add_trace(
            go.Scatter(
                x=df_chart["EOM_Str"],
                y=df_chart["Adj_R_Squared"],
                name="Model Fit Metric",
                mode="lines+markers",
                line=dict(color="#00FF41", width=4),
                marker=dict(size=8, color="#00FF41")
            )
        )
        fig_r2.update_layout(
            xaxis_title="End of Month (EOM)",
            yaxis_title="Fit Score / Adj R-Squared",
            yaxis=dict(range=[0, 1.05]),
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig_r2, use_container_width=True)

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
        st.info("👈 Select your model in the sidebar and click **Run Attribution Model** to begin.")

# =============================================================================
# TOOL 2: SAA PORTFOLIO OPTIMIZER
# =============================================================================
elif app_mode == "SAA Portfolio Optimizer":
    st.title("📈 ASX Vanguard Strategic Asset Allocation (SAA) Optimizer")
    st.markdown("Select your preferred Vanguard ASX ETFs, set your risk-free rate, and calculate the efficient frontier using 20 years of historical market data.")

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
            
            with st.spinner("Fetching 10 years of historical data from Yahoo Finance and simulating 5,000 portfolios..."):
                data = yf.download(selected_etfs, period="20y", auto_adjust=False, progress=False)
                
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
                        title='<b>Efficient Frontier & Capital Market Line (10-Year Historical Data)</b>',
                        xaxis_title='Annualized Volatility (Standard Deviation)',
                        yaxis_title='Annualized Return',
                        template='plotly_dark',
                        hovermode='closest'
                    )
                    
                    st.success("Optimization completed successfully with 10 years of history!")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 🎯 Optimal Portfolio Metrics & Allocation")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Expected Annual Return", f"{optimal_return*100:.2f}%")
                    col2.metric("Annualized Volatility", f"{optimal_volatility*100:.2f}%")
                    col3.metric("Sharpe Ratio", f"{optimal_sharpe:.2f}")
                    
                    etf_table_data = []
                    for t in selected_etfs:
                        if isinstance(prices, pd.DataFrame):
                            p_series = prices[t].dropna()
                        else:
                            p_series = prices.dropna()
                            
                        start_date = p_series.index.min().strftime('%Y-%m-%d')
                        end_date = p_series.index.max().strftime('%Y-%m-%d')
                        period_str = f"{start_date} to {end_date}"
                        
                        t_returns = daily_returns[t] if isinstance(daily_returns, pd.DataFrame) else daily_returns
                        ann_ret = t_returns.mean() * 252
                        ann_std = t_returns.std() * np.sqrt(252)
                        
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

# =============================================================================
# TOOL 3: FIXED INCOME RETURN ATTRIBUTION
# =============================================================================
elif app_mode == "Fixed Income Attribution":
    st.title("🛡️ Fixed Income Return Attribution")
    st.markdown("Attribute fixed income returns into running yield, yield curve impact, credit spread impact, and residual components.")

    st.sidebar.header("Fixed Income Parameters")
    st.sidebar.markdown("Janus Henderson Australian Fixed Interest Fund Defaults:")

    sb_col1, sb_col2 = st.sidebar.columns(2)

    with sb_col1:
        st.markdown("### **Fund (Portfolio)**")
        p_dur = st.number_input("Mod Duration", value=5.32, step=0.01, format="%.2f")
        p_yield = st.number_input("Current Yield (%)", value=5.24, step=0.01, format="%.2f")
        p_cs_dur = st.number_input("Spread Duration", value=1.47, step=0.01, format="%.2f")
        p_ret = st.number_input("1M Return (%)", value=-0.94, step=0.01, format="%.2f")

    with sb_col2:
        st.markdown("### **Index (Benchmark)**")
        b_dur = st.number_input("Mod Duration ", value=4.75, step=0.01, format="%.2f")
        b_yield = st.number_input("Current Yield (%) ", value=3.63, step=0.01, format="%.2f")
        b_cs_dur = st.number_input("Spread Duration ", value=0.46, step=0.01, format="%.2f")
        b_ret = st.number_input("1M Return (%) ", value=-0.86, step=0.01, format="%.2f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### **Market Movements**")
    m_col1, m_col2 = st.sidebar.columns(2)
    with m_col1:
        y_curr = st.number_input("CGB 5Y (Cur ME)", value=4.995, step=0.001, format="%.3f")
        cs_curr = st.number_input("Cred Spd (Cur ME)", value=0.905, step=0.001, format="%.3f")
    with m_col2:
        y_prior = st.number_input("CGB 5Y (Pri ME)", value=4.738, step=0.001, format="%.3f")
        cs_prior = st.number_input("Cred Spd (Pri ME)", value=0.812, step=0.001, format="%.3f")

    run_fi = st.sidebar.button("Run Fixed Income Attribution", type="primary")

    if run_fi:
        delta_y = y_curr - y_prior
        delta_cs = cs_curr - cs_prior

        # Portfolio (Fund) Attribution Components
        p_run_yield = p_yield / 12.0
        p_yc_impact = -p_dur * delta_y
        p_cs_impact = -p_cs_dur * delta_cs
        p_sum_known = p_run_yield + p_yc_impact + p_cs_impact
        p_residual = p_ret - p_sum_known

        # Benchmark (Index) Attribution Components
        b_run_yield = b_yield / 12.0
        b_yc_impact = -b_dur * delta_y
        b_cs_impact = -b_cs_dur * delta_cs
        b_sum_known = b_run_yield + b_yc_impact + b_cs_impact
        b_residual = b_ret - b_sum_known

        # Active Variances
        diff_run_yield = p_run_yield - b_run_yield
        diff_yc = p_yc_impact - b_yc_impact
        diff_cs = p_cs_impact - b_cs_impact
        diff_res = p_residual - b_residual

        # --- WATERFALL CHART ---
        waterfall_x = [
            "Index Return",
            "Running Yield Var",
            "Yield Curve Var",
            "Credit Spread Var",
            "Residual Var",
            "Fund Return"
        ]
        waterfall_y = [
            b_ret,
            diff_run_yield,
            diff_yc,
            diff_cs,
            diff_res,
            p_ret
        ]
        waterfall_measure = [
            "absolute",
            "relative",
            "relative",
            "relative",
            "relative",
            "total"
        ]

        fig_waterfall = go.Figure(go.Waterfall(
            name="Fixed Income Attribution",
            orientation="v",
            measure=waterfall_measure,
            x=waterfall_x,
            textposition="outside",
            text=[f"{float(val):+.2f}%" if m != "absolute" and m != "total" else f"{float(val):.2f}%" for m, val in zip(waterfall_measure, waterfall_y)],
            y=waterfall_y,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            increasing={"marker": {"color": "#2ecc71"}},
            totals={"marker": {"color": "#3498db"}}
        ))

        fig_waterfall.update_layout(
            title="<b>Fixed Income Return Attribution Waterfall (Index to Fund)</b>",
            yaxis_title="Return (%)",
            template="plotly_dark",
            showlegend=False
        )

        st.subheader("📊 Attribution Waterfall Chart")
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.markdown("---")

        # --- SUMMARY METRICS (PLACED BEFORE THE TABLE) ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Fund Excess Return Over Index", f"{p_ret - b_ret:.2f}%")
        col2.metric("Fund Total Residual", f"{p_residual:.2f}%")
        col3.metric("Index Total Residual", f"{b_residual:.2f}%")

        # --- ATTRIBUTION RESULTS TABLE ---
        attribution_table = pd.DataFrame({
            "Attribution Component": [
                "Running Yield Contribution",
                "Yield Curve Impact",
                "Credit Spread Impact",
                "Sum of Attributed",
                "Actual 1-Month Return",
                "Residual (Idiosyncratic)"
            ],
            "Fund (%)": [
                round(p_run_yield, 4),
                round(p_yc_impact, 4),
                round(p_cs_impact, 4),
                round(p_sum_known, 4),
                round(p_ret, 4),
                round(p_residual, 4)
            ],
            "Index (%)": [
                round(b_run_yield, 4),
                round(b_yc_impact, 4),
                round(b_cs_impact, 4),
                round(b_sum_known, 4),
                round(b_ret, 4),
                round(b_residual, 4)
            ],
            "Active Spread (Fund - Index) (%)": [
                round(diff_run_yield, 4),
                round(diff_yc, 4),
                round(diff_cs, 4),
                round(p_sum_known - b_sum_known, 4),
                round(p_ret - b_ret, 4),
                round(diff_res, 4)
            ]
        })

        st.subheader("📋 Fixed Income Return Attribution Results Table")
        st.markdown(f"**Period Evaluated:** 2026-08-31 to 2026-09-30 (ΔYield: {delta_y:+.3f}%, ΔCredit Spread: {delta_cs:+.3f}%)")
        st.dataframe(attribution_table, use_container_width=True)
    else:
        st.info("👈 Review the default Janus Henderson Fund parameters in the sidebar, then click **Run Fixed Income Attribution**.")
