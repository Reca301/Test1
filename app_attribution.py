import io
import pandas as pd
import statsmodels.api as sm
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Brigher Super Australian Shares - Accumulation account Factor Attribution",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Australian Shares Accumulation Expanding-Window Factor Attribution")
st.markdown("This app reads data from your repository folder (`Test1/AGX1_test.xlsx`), runs expanding-window OLS regressions, and displays the attribution results.")

# File path referencing your Test1 folder in GitHub
FILE_PATH = "data/BSAUS_test.xlsx"

@st.cache_data
def run_attribution(path):
    required_cols = ["EOM", "BSAUS_BM", "BM_RBA", "SMB", "HML", "UMD"]
    
    # Read Excel file from Test1 folder
    df_raw = (
        pd.read_excel(path, sheet_name="Data")[required_cols]
        .dropna()
        .sort_values("EOM")
        .reset_index(drop=True)
    )
    df_raw["EOM"] = pd.to_datetime(df_raw["EOM"])

    attribution_results = []

    # Expanding-window regression starting at 36th observation (index 35)
    for i in range(35, len(df_raw)):
        df_hist = df_raw.iloc[: i + 1].copy()
        y = df_hist["BSAUS_BM"]
        X = sm.add_constant(df_hist[["BM_RBA", "SMB", "HML", "UMD"]])
        model = sm.OLS(y, X).fit()

        alpha_active = model.params["const"]
        beta_active = model.params["BM_RBA"]
        size_p = model.params["SMB"]
        value_p = model.params["HML"]
        momentum_p = model.params["UMD"]

        current_row = df_raw.iloc[i]
        agx1_return = current_row["BSAUS_BM"]
        msci_rba = current_row["BM_RBA"]
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
            "BSAUS_BM": agx1_return,
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

# --- UI Button & Output Panel ---
if st.button("Run Factor Attribution", type="primary"):
    with st.spinner("Processing expanding-window regressions..."):
        df_attribution = run_attribution(FILE_PATH)

    st.success("Attribution calculations completed successfully!")

    # Display main results table
    st.subheader("📋 Attribution Results Table")
    st.dataframe(df_attribution, use_container_width=True)

    # Display reconciliation check
    st.subheader("🔍 Reconciliation Check (Last 10 Observations)")
    st.dataframe(
        df_attribution[["EOM", "BSAUS_BM", "Reconstructed_Return"]].tail(10),
        use_container_width=True,
    )

    # Excel Download Button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_attribution.to_excel(writer, sheet_name="Attribution", index=False)

    st.download_button(
        label="📥 Download Attribution Results as Excel",
        data=buffer.getvalue(),
        file_name="BSAUS_Attribution_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("👈 Click **Run Factor Attribution** to load your dataset from `data/BSAUS_test.xlsx` and view the web output.")
