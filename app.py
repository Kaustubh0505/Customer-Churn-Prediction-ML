import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 13px; color: #64748B; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #1E293B; }

[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E2E8F0;
}

.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(99,102,241,0.4);
}

h1 { font-weight: 700; color: #1E293B; }
h2, h3 { font-weight: 600; color: #334155; }

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6366F1;
    border-bottom-color: #6366F1;
}
</style>
""", unsafe_allow_html=True)

PALETTE = {
    "churn":   "#F43F5E",
    "stay":    "#10B981",
    "accent1": "#6366F1",
    "accent2": "#8B5CF6",
    "accent3": "#F59E0B",
    "accent4": "#3B82F6",
    "accent5": "#EC4899",
}

CHART_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13, color="#334155"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=20, b=20, l=10, r=10),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter"),
)

COLOR_LABEL = {PALETTE["churn"]: "Churn", PALETTE["stay"]: "No Churn"}
DISCRETE_CHURN = {"Churn": PALETTE["churn"], "No Churn": PALETTE["stay"]}

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df

model  = load_model()
raw_df = load_data()

ENCODINGS = {
    "gender":           {"Female": 0, "Male": 1},
    "Partner":          {"No": 0, "Yes": 1},
    "Dependents":       {"No": 0, "Yes": 1},
    "PhoneService":     {"No": 0, "Yes": 1},
    "MultipleLines":    {"No": 0, "No phone service": 1, "Yes": 2},
    "InternetService":  {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity":   {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup":     {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV":      {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies":  {"No": 0, "No internet service": 1, "Yes": 2},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod":    {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3,
    },
}

def encode(field, value):
    return ENCODINGS[field][value]

def encode_df(df):
    d = df.copy()
    for col, mapping in ENCODINGS.items():
        if col in d.columns:
            d[col] = d[col].map(mapping)
    d["gender"]        = d["gender"].map({"Female": 0, "Male": 1})
    d["SeniorCitizen"] = d["SeniorCitizen"].astype(int)
    return d

def batch_predict(df):
    feature_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    enc  = encode_df(df)
    X    = enc[feature_cols]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return preds, probs

def styled_chart(fig):
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", linewidth=1)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", linewidth=1)
    return fig

with st.sidebar:
    st.markdown("### 🔧 Dashboard Filters")
    st.caption("All charts update in real-time based on your selection.")
    st.divider()

    contract_opts = ["All"] + sorted(raw_df["Contract"].unique().tolist())
    sel_contract  = st.selectbox("Contract Type", contract_opts)

    internet_opts = ["All"] + sorted(raw_df["InternetService"].unique().tolist())
    sel_internet  = st.selectbox("Internet Service", internet_opts)

    senior_opts = ["All", "Senior Citizen", "Non-Senior"]
    sel_senior  = st.selectbox("Senior Citizen", senior_opts)

    tenure_range  = st.slider("Tenure (months)", 0, 72, (0, 72))
    monthly_range = st.slider(
        "Monthly Charges ($)",
        float(raw_df["MonthlyCharges"].min()),
        float(raw_df["MonthlyCharges"].max()),
        (float(raw_df["MonthlyCharges"].min()), float(raw_df["MonthlyCharges"].max())),
    )

filtered_df = raw_df.copy()
if sel_contract != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == sel_contract]
if sel_internet != "All":
    filtered_df = filtered_df[filtered_df["InternetService"] == sel_internet]
if sel_senior == "Senior Citizen":
    filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 1]
elif sel_senior == "Non-Senior":
    filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 0]
filtered_df = filtered_df[
    (filtered_df["tenure"] >= tenure_range[0]) &
    (filtered_df["tenure"] <= tenure_range[1]) &
    (filtered_df["MonthlyCharges"] >= monthly_range[0]) &
    (filtered_df["MonthlyCharges"] <= monthly_range[1])
]

preds, probs = batch_predict(filtered_df)
filtered_df  = filtered_df.copy()
filtered_df["Churn_Prediction"]  = preds
filtered_df["Churn_Probability"] = probs
filtered_df["Churn_Label"]       = filtered_df["Churn_Prediction"].map({1: "Churn", 0: "No Churn"})

st.markdown("# 📊 Customer Churn Intelligence")
st.markdown("Real-time churn analytics and individual risk assessment powered by Machine Learning.")

tab1, tab2 = st.tabs(["📈  Dashboard & Analytics", "🔍  Single Customer Prediction"])

with tab1:

    total      = len(filtered_df)
    churned    = int(filtered_df["Churn_Prediction"].sum())
    stayed     = total - churned
    churn_rate = (churned / total * 100) if total > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Customers Analysed", f"{total:,}")
    k2.metric("🔴 Predicted to Churn",  f"{churned:,}")
    k3.metric("🟢 Predicted to Stay",   f"{stayed:,}")
    k4.metric("📉 Churn Rate",          f"{churn_rate:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Churn vs Retention")
        labels = ["Churn", "No Churn"]
        values = [churned, stayed]
        fig_pie = go.Figure(go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=[PALETTE["churn"], PALETTE["stay"]], line=dict(color="white", width=3)),
            textinfo="percent",
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px'>Churn Rate</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#1E293B"), align="center",
        )
        fig_pie.update_layout(**CHART_LAYOUT, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Churn Rate by Contract Type")
        ct = filtered_df.groupby("Contract").apply(
            lambda x: pd.Series({
                "Churn Rate (%)": round(x["Churn_Prediction"].mean() * 100, 1),
                "Count": len(x),
            })
        ).reset_index()
        fig_ct = go.Figure(go.Bar(
            x=ct["Contract"],
            y=ct["Churn Rate (%)"],
            marker=dict(
                color=ct["Churn Rate (%)"],
                colorscale=[[0, PALETTE["stay"]], [0.5, PALETTE["accent3"]], [1, PALETTE["churn"]]],
                showscale=True,
                colorbar=dict(title="Churn %", thickness=12),
            ),
            text=ct["Churn Rate (%)"].map(lambda v: f"{v}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata:,}<extra></extra>",
            customdata=ct["Count"],
        ))
        styled_chart(fig_ct)
        fig_ct.update_layout(yaxis_title="Churn Rate (%)", xaxis_title="")
        st.plotly_chart(fig_ct, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Churn Distribution by Tenure Group")
        filtered_df["Tenure Group"] = pd.cut(
            filtered_df["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=["0–12m", "13–24m", "25–36m", "37–48m", "49–60m", "61–72m"],
            right=True,
        )
        tg = (
            filtered_df.groupby(["Tenure Group", "Churn_Label"], observed=True)
            .size().reset_index(name="Count")
        )
        fig_tg = px.bar(
            tg, x="Tenure Group", y="Count", color="Churn_Label",
            barmode="stack",
            color_discrete_map=DISCRETE_CHURN,
            text="Count",
        )
        fig_tg.update_traces(texttemplate="%{text}", textposition="inside", textfont_size=11)
        fig_tg.update_layout(**CHART_LAYOUT, legend_title_text="", xaxis_title="", yaxis_title="Customers")
        fig_tg.update_xaxes(showgrid=False, linecolor="#E2E8F0")
        fig_tg.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
        st.plotly_chart(fig_tg, use_container_width=True)

    with c4:
        st.markdown("#### Monthly Charges vs Churn Probability")
        sample = filtered_df.sample(min(1500, len(filtered_df)), random_state=42)
        fig_sc = px.scatter(
            sample,
            x="MonthlyCharges",
            y="Churn_Probability",
            color="Churn_Label",
            color_discrete_map=DISCRETE_CHURN,
            opacity=0.55,
            size_max=8,
            trendline="lowess",
            trendline_scope="overall",
            trendline_color_override=PALETTE["accent1"],
            labels={"Churn_Probability": "Churn Probability", "MonthlyCharges": "Monthly Charges ($)"},
        )
        fig_sc.update_traces(marker=dict(size=6))
        fig_sc.update_layout(**CHART_LAYOUT, legend_title_text="")
        fig_sc.update_xaxes(showgrid=False, linecolor="#E2E8F0")
        fig_sc.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickformat=".0%")
        st.plotly_chart(fig_sc, use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        st.markdown("#### Churn by Internet Service")
        isc = (
            filtered_df.groupby(["InternetService", "Churn_Label"])
            .size().reset_index(name="Count")
        )
        fig_isc = px.bar(
            isc, x="InternetService", y="Count", color="Churn_Label",
            barmode="group",
            color_discrete_map=DISCRETE_CHURN,
            text="Count",
        )
        fig_isc.update_traces(texttemplate="%{text:,}", textposition="outside", textfont_size=11)
        fig_isc.update_layout(**CHART_LAYOUT, legend_title_text="", xaxis_title="", yaxis_title="Customers")
        fig_isc.update_xaxes(showgrid=False, linecolor="#E2E8F0")
        fig_isc.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0")
        st.plotly_chart(fig_isc, use_container_width=True)

    with c6:
        st.markdown("#### Churn Risk Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x="Churn_Probability",
            color="Churn_Label",
            nbins=40,
            opacity=0.8,
            color_discrete_map=DISCRETE_CHURN,
            labels={"Churn_Probability": "Churn Probability"},
            barmode="overlay",
        )
        fig_hist.update_layout(**CHART_LAYOUT, legend_title_text="")
        fig_hist.update_xaxes(tickformat=".0%", showgrid=False, linecolor="#E2E8F0")
        fig_hist.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.markdown("#### 📋 Full Prediction Table")

    show_df = filtered_df[[
        "customerID", "tenure", "Contract", "MonthlyCharges",
        "Churn_Label", "Churn_Prediction", "Churn_Probability"
    ]].copy()
    show_df["Churn_Probability"] = (show_df["Churn_Probability"] * 100).map("{:.2f}%".format)

    def colour_label(val):
        c = "#F43F5E" if val == "Churn" else "#10B981"
        return f"color: {c}; font-weight: 600"

    st.dataframe(
        show_df.style.applymap(colour_label, subset=["Churn_Label"]),
        use_container_width=True,
        height=360,
    )

    st.divider()
    st.markdown("#### ⚠️ High-Risk Customers — Churn Probability ≥ 70%")

    high_risk = filtered_df[filtered_df["Churn_Probability"] >= 0.70].copy()
    high_risk_sorted = high_risk.sort_values("Churn_Probability", ascending=False)
    high_risk_sorted["Churn_Probability"] = (high_risk_sorted["Churn_Probability"] * 100).map("{:.2f}%".format)
    high_risk_disp = high_risk_sorted[[
        "customerID", "Churn_Label", "Churn_Probability", "Contract", "tenure", "MonthlyCharges"
    ]]

    if high_risk_disp.empty:
        st.info("No high-risk customers found with the current filters.")
    else:
        st.dataframe(
            high_risk_disp.style.applymap(colour_label, subset=["Churn_Label"]),
            use_container_width=True,
            height=320,
        )

with tab2:

    st.markdown("#### 🔍 Predict Churn for a Single Customer")
    st.markdown("Fill in the customer profile below and hit **Predict** to get an instant risk assessment.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        gender         = st.selectbox("Gender",         ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Partner",        ["No", "Yes"])
        dependents     = st.selectbox("Dependents",     ["No", "Yes"])

    with col2:
        st.markdown("**📱 Services**")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        phone_service   = st.selectbox("Phone Service",   ["No", "Yes"])
        multiple_lines  = st.selectbox("Multiple Lines",  ["No", "No phone service", "Yes"])
        internet_svc    = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
        online_backup   = st.selectbox("Online Backup",   ["No", "No internet service", "Yes"])

    with col3:
        st.markdown("**💳 Account**")
        device_prot    = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
        tech_support   = st.selectbox("Tech Support",      ["No", "No internet service", "Yes"])
        streaming_tv   = st.selectbox("Streaming TV",      ["No", "No internet service", "Yes"])
        streaming_mov  = st.selectbox("Streaming Movies",  ["No", "No internet service", "Yes"])
        contract       = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
        paperless_bill = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check",
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total Charges ($)",   0.0, 10000.0, 1000.0, step=10.0)

    st.divider()

    if st.button("🔍  Predict Churn Risk", use_container_width=True, type="primary"):
        features = np.array([[
            encode("gender", gender),
            1 if senior_citizen == "Yes" else 0,
            encode("Partner", partner),
            encode("Dependents", dependents),
            tenure,
            encode("PhoneService", phone_service),
            encode("MultipleLines", multiple_lines),
            encode("InternetService", internet_svc),
            encode("OnlineSecurity", online_security),
            encode("OnlineBackup", online_backup),
            encode("DeviceProtection", device_prot),
            encode("TechSupport", tech_support),
            encode("StreamingTV", streaming_tv),
            encode("StreamingMovies", streaming_mov),
            encode("Contract", contract),
            encode("PaperlessBilling", paperless_bill),
            encode("PaymentMethod", payment_method),
            monthly_charges,
            total_charges,
        ]])

        prediction  = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        churn_prob  = probability[1] * 100
        stay_prob   = probability[0] * 100

        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            if prediction == 1:
                st.error("⚠️  **This customer is likely to CHURN.**  Act now to retain them.")
            else:
                st.success("✅  **This customer is likely to STAY.**  Engagement looks healthy.")
        with r2:
            st.metric("🔴 Churn Probability",     f"{churn_prob:.1f}%")
        with r3:
            st.metric("🟢 Retention Probability", f"{stay_prob:.1f}%")

        bar_color = PALETTE["churn"] if churn_prob >= 50 else PALETTE["stay"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob,
            delta={"reference": 26.5, "suffix": "% vs avg"},
            number={"suffix": "%", "font": {"size": 36, "color": "#1E293B"}},
            title={"text": "Churn Risk Score", "font": {"size": 16, "color": "#64748B"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar":  {"color": bar_color, "thickness": 0.25},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "#DCFCE7"},
                    {"range": [40, 70], "color": "#FEF9C3"},
                    {"range": [70, 100],"color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 3},
                    "thickness": 0.75,
                    "value": churn_prob,
                },
            },
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(t=40, b=20, l=30, r=30),
            paper_bgcolor="white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### 📝 Customer Profile Summary")
        s1, s2, s3 = st.columns(3)
        profile = {
            "Gender": gender, "Senior Citizen": senior_citizen,
            "Partner": partner, "Dependents": dependents,
            "Tenure": f"{tenure} months", "Contract": contract,
            "Internet Service": internet_svc, "Phone Service": phone_service,
            "Monthly Charges": f"${monthly_charges:.2f}",
            "Total Charges": f"${total_charges:.2f}",
            "Churn Prediction": "🔴 Churn" if prediction == 1 else "🟢 No Churn",
            "Churn Probability": f"{churn_prob:.1f}%",
        }
        items = list(profile.items())
        for i, (k, v) in enumerate(items):
            col = [s1, s2, s3][i % 3]
            col.markdown(f"**{k}:** {v}")
