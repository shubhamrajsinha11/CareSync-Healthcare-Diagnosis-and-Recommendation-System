"""
Analytics Dashboard Utilities
Clean, readable health visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(28,28,28,1)",
    font=dict(color="#A0A0A0", size=13),
    margin=dict(l=60, r=40, t=70, b=60),
)
_GRID = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.06)",
    zerolinecolor="rgba(255,255,255,0.1)",
)


# ── Helpers ────────────────────────────────────────────────────────────────

def calculate_bmi(weight_kg, height_m):
    if weight_kg > 0 and height_m > 0:
        return round(weight_kg / (height_m ** 2), 1)
    return None

def get_bmi_category(bmi):
    if bmi is None:  return "Unknown"
    if bmi < 18.5:   return "Underweight"
    if bmi < 25.0:   return "Normal"
    if bmi < 30.0:   return "Overweight"
    return "Obese"

def get_bmi_color(bmi):
    if bmi is None:  return "#A0A0A0"
    if bmi < 18.5:   return "#60AEFF"
    if bmi < 25.0:   return "#4ADE80"
    if bmi < 30.0:   return "#FBBF24"
    return "#F87171"


# ── 1. Health Snapshot ─────────────────────────────────────────────────────

def create_health_metrics_dashboard(bmi, temperature, symptoms_count, risk_score):
    bmi_cat   = get_bmi_category(bmi)
    bmi_color = get_bmi_color(bmi)

    if temperature and temperature < 36.1:
        temp_status, temp_color = "Low", "#60AEFF"
    elif temperature and temperature <= 37.2:
        temp_status, temp_color = "Normal", "#4ADE80"
    elif temperature and temperature <= 38.0:
        temp_status, temp_color = "Mild Fever", "#FBBF24"
    else:
        temp_status, temp_color = "High Fever", "#F87171"

    risk_pct = round((risk_score or 0) * 100, 1)
    if risk_pct < 40:
        risk_status, risk_color = "Low Risk", "#4ADE80"
    elif risk_pct < 70:
        risk_status, risk_color = "Moderate Risk", "#FBBF24"
    else:
        risk_status, risk_color = "High Risk", "#F87171"

    sym_color = "#4ADE80" if symptoms_count <= 3 else "#FBBF24" if symptoms_count <= 6 else "#F87171"

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        vertical_spacing=0.25,
        horizontal_spacing=0.15,
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=bmi or 0,
        title={"text": f"<b>BMI</b><br><span style='font-size:13px;color:{bmi_color}'>{bmi_cat}</span>",
               "font": {"size": 16, "color": "#F0F0F0"}},
        number={"font": {"size": 36, "color": "#F0F0F0"}},
        gauge={
            "axis": {"range": [10, 40], "tickcolor": "#606060",
                     "tickvals": [18.5, 25, 30], "ticktext": ["18.5", "25", "30"]},
            "bar": {"color": bmi_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [10,   18.5], "color": "rgba(96,170,255,0.15)"},
                {"range": [18.5, 25],   "color": "rgba(74,222,128,0.15)"},
                {"range": [25,   30],   "color": "rgba(251,191,36,0.15)"},
                {"range": [30,   40],   "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {"line": {"color": bmi_color, "width": 3},
                          "thickness": 0.8, "value": bmi or 0},
        },
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=temperature or 0,
        title={"text": f"<b>Body Temperature</b><br><span style='font-size:13px;color:{temp_color}'>{temp_status}</span>",
               "font": {"size": 16, "color": "#F0F0F0"}},
        number={"font": {"size": 36, "color": "#F0F0F0"}, "suffix": " °C"},
        gauge={
            "axis": {"range": [34, 42], "tickcolor": "#606060",
                     "tickvals": [36.1, 37.2, 38.0], "ticktext": ["36.1", "37.2", "38"]},
            "bar": {"color": temp_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [34,   36.1], "color": "rgba(96,170,255,0.15)"},
                {"range": [36.1, 37.2], "color": "rgba(74,222,128,0.15)"},
                {"range": [37.2, 38.0], "color": "rgba(251,191,36,0.15)"},
                {"range": [38.0, 42.0], "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {"line": {"color": temp_color, "width": 3},
                          "thickness": 0.8, "value": temperature or 0},
        },
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="number",
        value=symptoms_count or 0,
        title={"text": "<b>Symptoms Reported</b><br><span style='font-size:13px;color:#A0A0A0'>in this session</span>",
               "font": {"size": 16, "color": "#F0F0F0"}},
        number={"font": {"size": 52, "color": sym_color}, "suffix": " symptoms"},
    ), row=2, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        title={"text": f"<b>Overall Health Risk</b><br><span style='font-size:13px;color:{risk_color}'>{risk_status}</span>",
               "font": {"size": 16, "color": "#F0F0F0"}},
        number={"font": {"size": 36, "color": "#F0F0F0"}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#606060",
                     "tickvals": [0, 40, 70, 100],
                     "ticktext": ["0", "40%", "70%", "100%"]},
            "bar": {"color": risk_color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "rgba(74,222,128,0.15)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                {"range": [70,100], "color": "rgba(248,113,113,0.15)"},
            ],
            "threshold": {"line": {"color": risk_color, "width": 3},
                          "thickness": 0.8, "value": risk_pct},
        },
    ), row=2, col=2)

    fig.update_layout(
        height=580,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0A0A0"),
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


# ── 2. Symptom Frequency ───────────────────────────────────────────────────

def create_symptom_frequency_chart(symptom_data):
    if not symptom_data:
        return None

    df     = pd.DataFrame(symptom_data)
    counts = df["symptom"].value_counts().head(10).sort_values()
    max_c  = counts.max()

    # Colour intensity based on frequency — more frequent = deeper red
    bar_colors = [
        f"rgba(248,113,113,{round(0.35 + 0.65 * (v / max_c), 2)})"
        for v in counts.values
    ]

    symptom_labels = [s.replace("_", " ").title() for s in counts.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts.values,
        y=symptom_labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(248,113,113,0.3)", width=1),
        ),
        text=[f" {v}x" for v in counts.values],
        textposition="outside",
        textfont=dict(color="#F0F0F0", size=12),
        hovertemplate="<b>%{y}</b><br>Reported <b>%{x}</b> time(s)<extra></extra>",
    ))

    # Average line
    if len(counts) > 1:
        avg = counts.mean()
        fig.add_vline(
            x=avg,
            line_dash="dot",
            line_color="rgba(251,191,36,0.5)",
            line_width=1.5,
            annotation_text=f"avg ({avg:.1f})",
            annotation_font_color="#FBBF24",
            annotation_font_size=10,
            annotation_position="top",
        )

    fig.update_layout(
        title=dict(
            text="Symptom Frequency",
            font=dict(size=16, color="#F0F0F0"), x=0,
        ),
        xaxis=dict(title="Times Reported", **_GRID, color="#606060", tickformat="d"),
        yaxis=dict(
            title="",
            color="#F0F0F0",
            tickfont=dict(size=12),
            showgrid=False,
        ),
        height=400,
        bargap=0.3,
        **_DARK,
    )
    return fig


# ── 3. BMI chart ───────────────────────────────────────────────────────────

def create_bmi_chart(bmi_history):
    if not bmi_history or len(bmi_history) < 2:
        return None

    df = pd.DataFrame(bmi_history)
    fig = go.Figure()

    for y0, y1, color, label in [
        (10,   18.5, "rgba(96,170,255,0.08)",  "Underweight"),
        (18.5, 25,   "rgba(74,222,128,0.08)",  "Normal"),
        (25,   30,   "rgba(251,191,36,0.08)",  "Overweight"),
        (30,   40,   "rgba(248,113,113,0.08)", "Obese"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="left",
                      annotation_font_color="rgba(200,200,200,0.45)",
                      annotation_font_size=10)

    bmi_colors = [get_bmi_color(b) for b in df["bmi"]]
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bmi"],
        mode="lines+markers",
        line=dict(color="#4A9EFF", width=2.5),
        marker=dict(size=12, color=bmi_colors, line=dict(color="#141414", width=2)),
        customdata=[[get_bmi_category(b)] for b in df["bmi"]],
        hovertemplate="<b>%{x}</b><br>BMI: %{y:.1f} (%{customdata[0]})<extra></extra>",
        name="BMI",
    ))

    fig.update_layout(
        title=dict(text="BMI Over Time  —  coloured zones show weight categories",
                   font=dict(size=15, color="#F0F0F0"), x=0),
        xaxis=dict(title="Date", **_GRID, color="#606060"),
        yaxis=dict(title="BMI", range=[10, 40], **_GRID, color="#606060"),
        height=400, showlegend=False, **_DARK,
    )
    return fig


# ── 4. Temperature chart ───────────────────────────────────────────────────

def create_temperature_chart(temperature_history):
    if not temperature_history or len(temperature_history) < 2:
        return None

    df = pd.DataFrame(temperature_history)
    fig = go.Figure()

    for y0, y1, color, label in [
        (34,   36.1, "rgba(96,170,255,0.08)",  "Low"),
        (36.1, 37.2, "rgba(74,222,128,0.08)",  "Normal"),
        (37.2, 38.0, "rgba(251,191,36,0.08)",  "Mild Fever"),
        (38.0, 42.0, "rgba(248,113,113,0.08)", "Fever"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="left",
                      annotation_font_color="rgba(200,200,200,0.45)",
                      annotation_font_size=10)

    temp_colors = [
        "#60AEFF" if t < 36.1 else
        "#4ADE80" if t <= 37.2 else
        "#FBBF24" if t <= 38.0 else
        "#F87171"
        for t in df["temperature"]
    ]
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["temperature"],
        mode="lines+markers",
        line=dict(color="#F87171", width=2.5),
        marker=dict(size=12, color=temp_colors, line=dict(color="#141414", width=2)),
        hovertemplate="<b>%{x}</b><br>Temperature: %{y:.1f} °C<extra></extra>",
        name="Temperature",
    ))

    fig.update_layout(
        title=dict(text="Body Temperature Over Time  —  normal range: 36.1 – 37.2 °C",
                   font=dict(size=15, color="#F0F0F0"), x=0),
        xaxis=dict(title="Date", **_GRID, color="#606060"),
        yaxis=dict(title="Temperature (°C)", range=[34, 42], **_GRID, color="#606060"),
        height=400, showlegend=False, **_DARK,
    )
    return fig


# ── 5. Risk Profile ────────────────────────────────────────────────────────

def create_disease_risk_chart(disease_risks):
    if not disease_risks:
        return None

    df          = pd.DataFrame(disease_risks)
    df["pct"]   = (df["risk_score"] * 100).round(1)
    df["color"] = df["pct"].apply(
        lambda v: "#4ADE80" if v < 40 else "#FBBF24" if v < 70 else "#F87171"
    )
    df["status"] = df["pct"].apply(
        lambda v: "Low Risk" if v < 40 else "Moderate Risk" if v < 70 else "High Risk"
    )
    df["advice"] = df["pct"].apply(lambda v:
        "You are in the safe zone. Monitor symptoms regularly." if v < 40 else
        "Caution advised. Consider seeing a doctor if symptoms persist." if v < 70 else
        "Please consult a healthcare professional as soon as possible."
    )

    fig = go.Figure()

    # Coloured zone backgrounds
    for x0, x1, color, label in [
        (0,  40,  "rgba(74,222,128,0.07)",  "Safe (0–40%)"),
        (40, 70,  "rgba(251,191,36,0.07)",  "Caution (40–70%)"),
        (70, 100, "rgba(248,113,113,0.07)", "Danger (70–100%)"),
    ]:
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=color, line_width=0,
            annotation_text=label,
            annotation_position="top",
            annotation_font_color="rgba(255,255,255,0.3)",
            annotation_font_size=10,
        )

    # Risk bar
    fig.add_trace(go.Bar(
        y=df["disease"],
        x=df["pct"],
        orientation="h",
        marker=dict(
            color=df["color"],
            opacity=0.85,
            line=dict(color="rgba(0,0,0,0.2)", width=1),
        ),
        text=[f"{p}%  —  {s}" for p, s in zip(df["pct"], df["status"])],
        textposition="inside",
        insidetextanchor="start",
        textfont=dict(color="white", size=13),
        customdata=list(zip(df["status"], df["advice"])),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Risk Score: <b>%{x:.1f}%</b><br>"
            "Status: <b>%{customdata[0]}</b><br>"
            "<i>%{customdata[1]}</i>"
            "<extra></extra>"
        ),
    ))

    # Threshold lines
    fig.add_vline(x=40, line_dash="dash", line_color="rgba(251,191,36,0.5)",
                  line_width=1.5)
    fig.add_vline(x=70, line_dash="dash", line_color="rgba(248,113,113,0.5)",
                  line_width=1.5)

    fig.update_layout(
        title=dict(
            text="Health Risk Profile  —  hover over the bar for advice",
            font=dict(size=15, color="#F0F0F0"), x=0,
        ),
        xaxis=dict(
            title="Risk Score (%)",
            range=[0, 105],
            tickvals=[0, 40, 70, 100],
            ticktext=["0%", "40%\n(Caution)", "70%\n(High)", "100%"],
            **_GRID, color="#606060",
        ),
        yaxis=dict(
            title="",
            color="#F0F0F0",
            tickfont=dict(size=13),
            showgrid=False,
        ),
        height=250,
        showlegend=False,
        **_DARK,
    )
    return fig


# ── 6. Trend analysis ──────────────────────────────────────────────────────

def create_trend_analysis(data_history):
    if not data_history or len(data_history) < 2:
        return None

    df = pd.DataFrame(data_history)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=("BMI Over Time", "Body Temperature Over Time"),
    )

    if "bmi" in df.columns:
        bmi_colors = [get_bmi_color(b) for b in df["bmi"]]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bmi"],
            mode="lines+markers",
            line=dict(color="#4A9EFF", width=2.5),
            marker=dict(size=9, color=bmi_colors, line=dict(color="#141414", width=2)),
            customdata=[[get_bmi_category(b)] for b in df["bmi"]],
            hovertemplate="<b>%{x}</b><br>BMI: %{y:.1f} (%{customdata[0]})<extra></extra>",
            name="BMI",
        ), row=1, col=1)

    if "temperature" in df.columns:
        t_colors = [
            "#60AEFF" if t < 36.1 else
            "#4ADE80" if t <= 37.2 else
            "#FBBF24" if t <= 38 else
            "#F87171"
            for t in df["temperature"]
        ]
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["temperature"],
            mode="lines+markers",
            line=dict(color="#F87171", width=2.5),
            marker=dict(size=9, color=t_colors, line=dict(color="#141414", width=2)),
            hovertemplate="<b>%{x}</b><br>Temp: %{y:.1f} °C<extra></extra>",
            name="Temperature",
        ), row=2, col=1)
        fig.add_hline(y=37.2, line_dash="dot",
                      line_color="rgba(74,222,128,0.4)", row=2, col=1)
        fig.add_hline(y=38.0, line_dash="dot",
                      line_color="rgba(248,113,113,0.4)", row=2, col=1)

    fig.update_xaxes(**_GRID, color="#606060")
    fig.update_yaxes(**_GRID, color="#606060")
    fig.update_yaxes(title_text="BMI",       row=1, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Date",      row=2, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = "#A0A0A0"
        ann.font.size  = 14

    fig.update_layout(
        height=580, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,28,28,1)",
        font=dict(color="#A0A0A0", size=13),
        margin=dict(l=60, r=40, t=50, b=50),
    )
    return fig