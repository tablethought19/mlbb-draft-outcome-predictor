import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import shap
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MLBB Draft Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Predict button — gold color */
    div.stButton > button[kind="primary"] {
        background-color: #00C853;
        color: #000000;
        font-weight: 700;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #00A846;
        color: #000000;
    }

    /* Global font size increase */
    html, body, [class*="css"] {
        font-size: 22px !important;
    }
    h1 { font-size: 2.4rem !important; }
    h2 { font-size: 2.0rem !important; }
    h3 { font-size: 1.6rem !important; }
    label, .stSelectbox label, .stMarkdown p {
        font-size: 22px !important;
    }
    div[data-baseweb="select"] {
        font-size: 22px !important;
    }
</style>
""", unsafe_allow_html=True)
# ── Load model and data ───────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/random_forest.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(_mdl):
    return shap.TreeExplainer(_mdl)

@st.cache_data
def load_data():
    X = pd.read_csv("data/processed/X.csv")
    df = pd.read_csv("data/raw/mlbb_drafts_clean.csv")
    return X, df

model      = load_model()
explainer  = load_explainer(model)
st.cache_data.clear()  
X_template, df = load_data()
FEATURE_COLS = X_template.columns.tolist()

# ── Hero list ─────────────────────────────────────────────────────
pick_cols = [c for c in FEATURE_COLS if c.startswith('pick_')]
HEROES = sorted([
    c.replace('pick_', '')
     .replace('_', ' ')
     .title()
     .replace("'S", "'s")
    for c in pick_cols
])

NAME_FIXES = {
    "Chang E":        "Chang'e",
    "Yi Sun Shin":    "Yi Sun-Shin",
    "X Borg":         "X.Borg",
    "Lapu Lapu":      "Lapu-Lapu",
    "Popol And Kupa": "Popol and Kupa",
    "Luo Yi":         "Luo Yi",
    "Yu Zhong":       "Yu Zhong",
}
HEROES = [NAME_FIXES.get(h, h) for h in HEROES]

# ── Patch list ────────────────────────────────────────────────────
PATCH_ORDER = [
    "1.8.30B", "1.8.56B","1.8.78", "1.8.92B",
    "1.9.06",  "1.9.06C", "1.9.42", "1.9.42E","1.9.64",
    "1.9.64B", "1.9.91",  "1.9.99", "1.9.99a","2.1.18",
    "2.1.40a", "2.1.61"
]
PATCH_MAP = {p: i for i, p in enumerate(PATCH_ORDER)}

# ── Hero name → feature col ───────────────────────────────────────
def hero_to_col(name):
    return str(name).lower()\
        .replace(" ", "_")\
        .replace("'", "")\
        .replace(".", "")\
        .replace("-", "_")

# ── Clean feature name for display ───────────────────────────────
def clean_feature_name(name, value):
    if name.startswith("pick_"):
        hero = name.replace("pick_", "").replace("_", " ").title()
        team = "Team 1" if value == 1 else "Team 2"
        return f"{hero} (picked by {team})"
    elif name.startswith("ban_"):
        hero = name.replace("ban_", "").replace("_", " ").title()
        team = "Team 1" if value == 1 else "Team 2"
        return f"{hero} (banned by {team})"
    elif name == "team1_is_blue":
        return "Blue side advantage"
    elif name == "patch_encoded":
        return "Patch version"
    return name

# ── Build input row ───────────────────────────────────────────────
def build_row(team1_picks, team2_picks, team1_bans, team2_bans, team1_is_blue, patch):
    row = {col: 0 for col in FEATURE_COLS}
    for hero in team1_picks:
        col = f"pick_{hero_to_col(hero)}"
        if col in row: row[col] = 1
    for hero in team2_picks:
        col = f"pick_{hero_to_col(hero)}"
        if col in row: row[col] = -1
    for hero in team1_bans:
        col = f"ban_{hero_to_col(hero)}"
        if col in row: row[col] = 1
    for hero in team2_bans:
        col = f"ban_{hero_to_col(hero)}"
        if col in row: row[col] = -1
    row['team1_is_blue'] = 1 if team1_is_blue else 0
    row['patch_encoded'] = PATCH_MAP.get(patch, max(PATCH_MAP.values()))
    return row

# ── Prediction function ───────────────────────────────────────────
def predict_draft(team1_picks, team2_picks, team1_bans, team2_bans, team1_is_blue, patch):
    row = build_row(team1_picks, team2_picks, team1_bans, team2_bans, team1_is_blue, patch)
    X_input = pd.DataFrame([row])[FEATURE_COLS]
    prob = model.predict_proba(X_input)[0]
    pred = model.predict(X_input)[0]
    return {
        "prediction":     "Team 1 wins" if pred == 1 else "Team 2 wins",
        "team1_win_prob": round(prob[1] * 100, 1),
        "team2_win_prob": round(prob[0] * 100, 1),
        "row":            row,
    }

# ── SHAP explanation ──────────────────────────────────────────────
def get_shap_explanation(row):
    X_input = pd.DataFrame([row])[FEATURE_COLS]
    shap_values = explainer.shap_values(X_input, check_additivity=False)
    vals = shap_values[0, :, 1]
    pairs = [
        (n, v, row[n]) for n, v in zip(FEATURE_COLS, vals)
        if abs(v) > 0.001 and row[n] != 0
    ]
    top_team1 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
    top_team2 = sorted(pairs, key=lambda x: x[1])[:5]
    return (
        [(clean_feature_name(n, val), v) for n, v, val in top_team1],
        [(clean_feature_name(n, val), v) for n, v, val in top_team2]
    )

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.title("MLBB Draft AI")
page = st.sidebar.radio("Navigation", ["Draft Predictor", "Meta Dashboard"])
st.sidebar.markdown("---")
st.sidebar.caption("Final Year Project | CS (AI)")
st.sidebar.markdown("---")
st.sidebar.caption("Data sourced from [Liquipedia](https://liquipedia.net) (CC-BY-SA 3.0)")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — DRAFT PREDICTOR
# ══════════════════════════════════════════════════════════════════
if page == "Draft Predictor":
    st.title("MLBB Draft Predictor")
    st.caption("Select heroes for both teams to predict the match outcome")
    st.markdown("---")

    col_side, col_patch, _ = st.columns([1, 1, 2])
    with col_side:
        side = st.selectbox("Team 1 Side", ["Blue", "Red"])
        team1_is_blue = side == "Blue"
    with col_patch:
        patch = st.selectbox("Patch", list(reversed(PATCH_ORDER)), index=0)

    st.markdown("---")

    col1, col_mid, col2 = st.columns([5, 1, 5])

    with col1:
            team1_color = "🔵" if team1_is_blue else "🔴"
            st.subheader(f"{team1_color} Team 1")

            def available_team1(current_key):
                current_val = st.session_state.get(current_key, "")
                used = set()
                for k in ["t1p1","t1p2","t1p3","t1p4","t1p5",
                        "t1b1","t1b2","t1b3","t1b4","t1b5"]:
                    if k != current_key:
                        v = st.session_state.get(k, "")
                        if v: used.add(v)
                opts = [""] + [h for h in HEROES if h not in used or h == current_val]
                return opts

            p1, b1 = st.columns(2)
            with p1:
                st.markdown("**Picks**")
                t1p1 = st.selectbox("Pick 1", available_team1("t1p1"), key="t1p1")
                t1p2 = st.selectbox("Pick 2", available_team1("t1p2"), key="t1p2")
                t1p3 = st.selectbox("Pick 3", available_team1("t1p3"), key="t1p3")
                t1p4 = st.selectbox("Pick 4", available_team1("t1p4"), key="t1p4")
                t1p5 = st.selectbox("Pick 5", available_team1("t1p5"), key="t1p5")
            with b1:
                st.markdown("**Bans**")
                t1b1 = st.selectbox("Ban 1", available_team1("t1b1"), key="t1b1")
                t1b2 = st.selectbox("Ban 2", available_team1("t1b2"), key="t1b2")
                t1b3 = st.selectbox("Ban 3", available_team1("t1b3"), key="t1b3")
                t1b4 = st.selectbox("Ban 4", available_team1("t1b4"), key="t1b4")
                t1b5 = st.selectbox("Ban 5", available_team1("t1b5"), key="t1b5")

    with col_mid:
        st.markdown("<br>" * 11, unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; font-size:28px; font-weight:900;'>VS</div>", unsafe_allow_html=True)

    with col2:
        team2_color = "🔴" if team1_is_blue else "🔵"
        st.subheader(f"{team2_color} Team 2")

        t1_selected = {t1p1, t1p2, t1p3, t1p4, t1p5,
                    t1b1, t1b2, t1b3, t1b4, t1b5} - {""}

        def available(current_key):
            current_val = st.session_state.get(current_key, "")
            used = t1_selected.copy()
            for k in ["t2p1","t2p2","t2p3","t2p4","t2p5",
                    "t2b1","t2b2","t2b3","t2b4","t2b5"]:
                if k != current_key:
                    v = st.session_state.get(k, "")
                    if v: used.add(v)
            opts = [""] + [h for h in HEROES if h not in used or h == current_val]
            return opts

        p2, b2 = st.columns(2)
        with p2:
            st.markdown("**Picks**")
            t2p1 = st.selectbox("Pick 1", available("t2p1"), key="t2p1")
            t2p2 = st.selectbox("Pick 2", available("t2p2"), key="t2p2")
            t2p3 = st.selectbox("Pick 3", available("t2p3"), key="t2p3")
            t2p4 = st.selectbox("Pick 4", available("t2p4"), key="t2p4")
            t2p5 = st.selectbox("Pick 5", available("t2p5"), key="t2p5")
        with b2:
            st.markdown("**Bans**")
            t2b1 = st.selectbox("Ban 1", available("t2b1"), key="t2b1")
            t2b2 = st.selectbox("Ban 2", available("t2b2"), key="t2b2")
            t2b3 = st.selectbox("Ban 3", available("t2b3"), key="t2b3")
            t2b4 = st.selectbox("Ban 4", available("t2b4"), key="t2b4")
            t2b5 = st.selectbox("Ban 5", available("t2b5"), key="t2b5")

    st.markdown("---")

    predict_btn = st.button("⚔️ Predict Winner", type="primary", use_container_width=True)

    if predict_btn:
        team1_picks = [h for h in [t1p1,t1p2,t1p3,t1p4,t1p5] if h]
        team2_picks = [h for h in [t2p1,t2p2,t2p3,t2p4,t2p5] if h]
        team1_bans  = [h for h in [t1b1,t1b2,t1b3,t1b4,t1b5] if h]
        team2_bans  = [h for h in [t2b1,t2b2,t2b3,t2b4,t2b5] if h]

        if len(team1_picks) < 5 or len(team2_picks) < 5:
            st.warning("Please select all 5 picks for both teams before predicting.")
        else:
            result = predict_draft(
                team1_picks, team2_picks,
                team1_bans, team2_bans,
                team1_is_blue, patch
            )

            st.markdown("---")
            st.markdown("## Prediction Result")

            res_col1, res_col2, res_col3 = st.columns([2, 1, 2])
            team1_prob = result['team1_win_prob']
            team2_prob = result['team2_win_prob']
            winner     = result['prediction']

            with res_col1:
                t1_color = "#1E90FF" if team1_is_blue else "#FF4444"
                if winner == "Team 1 wins":
                    st.markdown(f"""
                    <div style='text-align:center; background:{t1_color}22;
                    border: 2px solid {t1_color}; border-radius:12px; padding:20px'>
                        <div style='font-size:48px; font-weight:900; color:{t1_color}'>{team1_prob}%</div>
                        <div style='font-size:18px; color:{t1_color}; font-weight:600'>{'🔵' if team1_is_blue else '🔴'} Team 1</div>
                        <div style='font-size:14px; color:{t1_color}'>✓ PREDICTED WINNER</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='text-align:center; background:#33333322;
                    border: 2px solid #666; border-radius:12px; padding:20px'>
                        <div style='font-size:48px; font-weight:900; color:#aaa'>{team1_prob}%</div>
                        <div style='font-size:18px; color:#aaa'>{'🔵' if team1_is_blue else '🔴'} Team 1</div>
                    </div>""", unsafe_allow_html=True)

            with res_col2:
                st.markdown("<div style='text-align:center; font-size:28px; padding-top:30px'>VS</div>",
                            unsafe_allow_html=True)

            with res_col3:
                t2_color = "#FF4444" if team1_is_blue else "#1E90FF"
                if winner == "Team 2 wins":
                    st.markdown(f"""
                    <div style='text-align:center; background:{t2_color}22;
                    border: 2px solid {t2_color}; border-radius:12px; padding:20px'>
                        <div style='font-size:48px; font-weight:900; color:{t2_color}'>{team2_prob}%</div>
                        <div style='font-size:18px; color:{t2_color}; font-weight:600'>{'🔴' if team1_is_blue else '🔵'} Team 2</div>
                        <div style='font-size:14px; color:{t2_color}'>✓ PREDICTED WINNER</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='text-align:center; background:#33333322;
                    border: 2px solid #666; border-radius:12px; padding:20px'>
                        <div style='font-size:48px; font-weight:900; color:#aaa'>{team2_prob}%</div>
                        <div style='font-size:18px; color:#aaa'>{'🔴' if team1_is_blue else '🔵'} Team 2</div>
                    </div>""", unsafe_allow_html=True)

            # ── Probability bar ───────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            blue_prob = team1_prob if team1_is_blue else team2_prob
            red_prob  = team2_prob if team1_is_blue else team1_prob
            st.markdown(f"""
            <div style='margin-top:10px'>
                <div style='display:flex; height:24px; border-radius:12px; overflow:hidden'>
                    <div style='width:{blue_prob}%; background:#1E90FF;
                    display:flex; align-items:center; justify-content:center;
                    color:white; font-weight:700; font-size:13px'>🔵 {blue_prob}%</div>
                    <div style='width:{red_prob}%; background:#FF4444;
                    display:flex; align-items:center; justify-content:center;
                    color:white; font-weight:700; font-size:13px'>{red_prob}% 🔴</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── SHAP Explanation ──────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### 🔍 Why this prediction?")
            st.caption("Key factors that influenced the model's decision for this draft")

            team1_factors, team2_factors = get_shap_explanation(result['row'])

            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                t1_color = "#1E90FF" if team1_is_blue else "#FF4444"
                st.markdown(f"**{'🔵' if team1_is_blue else '🔴'} Factors favouring Team 1**")
                for name, val in team1_factors:
                    st.markdown(f"<span style='color:{t1_color}'>▲ {name}</span>",
                                unsafe_allow_html=True)

            with exp_col2:
                t2_color = "#FF4444" if team1_is_blue else "#1E90FF"
                st.markdown(f"**{'🔴' if team1_is_blue else '🔵'} Factors favouring Team 2**")
                for name, val in team2_factors:
                    st.markdown(f"<span style='color:{t2_color}'>▼ {name}</span>",
                                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("⚠️ Prediction is based on draft composition only. In-game performance, player skill, and strategy are not accounted for.")

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — META DASHBOARD
# ══════════════════════════════════════════════════════════════════
elif page == "Meta Dashboard":
    st.title("📊 Meta Dashboard")
    st.caption("Hero pick/ban rates and synergy analysis from competitive data")
    st.markdown("---")

    available_patches = sorted(df['patch'].dropna().unique())
    selected_patch = st.selectbox(
        "Filter by Patch",
        ["All Patches"] + list(reversed(available_patches))
    )

    df_filtered = df if selected_patch == "All Patches" else df[df['patch'] == selected_patch]
    st.caption(f"Showing data from {len(df_filtered)} games")
    st.markdown("---")

    pick_cols_raw = [f"team{t}pick{n}" for t in [1,2] for n in range(1,6)]
    ban_cols_raw  = [f"team{t}ban{n}"  for t in [1,2] for n in range(1,6)]

    all_picks = pd.concat([df_filtered[c] for c in pick_cols_raw]).dropna()
    all_bans  = pd.concat([df_filtered[c] for c in ban_cols_raw]).dropna()
    total_games = len(df_filtered)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗡️ Most Picked",
        "🚫 Most Banned",
        "🤝 Synergy",
        "📈 Patch Meta Shift",
        "⚔️ Counter Picks",
        "🏆 Win Rate"
    ])

    # ── Tab 1: Most Picked ────────────────────────────────────────
    with tab1:
        st.subheader("Most Picked Heroes")
        pick_counts = all_picks.value_counts().head(20).reset_index()
        pick_counts.columns = ['Hero', 'Picks']
        pick_counts['Pick Rate'] = (pick_counts['Picks'] / (total_games * 10) * 100).round(1)
        pick_counts['Hero'] = pick_counts['Hero'].str.title()

        fig = px.bar(
            pick_counts,
            x='Hero',
            y='Pick Rate',
            color_discrete_sequence=['#1E90FF'],
            labels={'Pick Rate': 'Pick Rate (%)'},
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            pick_counts.style.background_gradient(subset=['Pick Rate'], cmap='Blues'),
            use_container_width=True,
            hide_index=True
        )

    # ── Tab 2: Most Banned ────────────────────────────────────────
    with tab2:
        st.subheader("Most Banned Heroes")
        ban_counts = all_bans.value_counts().head(20).reset_index()
        ban_counts.columns = ['Hero', 'Bans']
        ban_counts['Ban Rate'] = (ban_counts['Bans'] / (total_games * 10) * 100).round(1)
        ban_counts['Hero'] = ban_counts['Hero'].str.title()

        fig2 = px.bar(
            ban_counts,
            x='Hero',
            y='Ban Rate',
            color_discrete_sequence=['#FF4444'],
            labels={'Ban Rate': 'Ban Rate (%)'},
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(
            ban_counts.style.background_gradient(subset=['Ban Rate'], cmap='Reds'),
            use_container_width=True,
            hide_index=True
        )

    # ── Tab 3: Synergy ────────────────────────────────────────────
    with tab3:
        st.subheader("Top Hero Synergies")
        st.caption("Hero pairs that appear together most often on the winning team")

        from itertools import combinations

        synergy_counts = {}
        synergy_games  = {}

        for _, row in df_filtered.iterrows():
            winner = row.get('winner')
            if str(winner) == '1':
                win_picks  = [row.get(f'team1pick{n}') for n in range(1,6)]
                lose_picks = [row.get(f'team2pick{n}') for n in range(1,6)]
            elif str(winner) == '2':
                win_picks  = [row.get(f'team2pick{n}') for n in range(1,6)]
                lose_picks = [row.get(f'team1pick{n}') for n in range(1,6)]
            else:
                continue

            win_picks = [h for h in win_picks if pd.notna(h)]
            for pair in combinations(sorted(win_picks), 2):
                synergy_counts[pair] = synergy_counts.get(pair, 0) + 1

            all_team_picks = win_picks + [h for h in lose_picks if pd.notna(h)]
            for pair in combinations(sorted(set(all_team_picks)), 2):
                synergy_games[pair] = synergy_games.get(pair, 0) + 1

        synergy_rows = []
        for pair, wins in synergy_counts.items():
            total = synergy_games.get(pair, wins)
            win_rate = round(wins / total * 100, 1) if total >= 10 else None
            synergy_rows.append({
                'Hero 1': pair[0].title(),
                'Hero 2': pair[1].title(),
                'Wins Together': wins,
                'Win Rate %': win_rate
            })

        synergy_df = pd.DataFrame(synergy_rows)\
            .sort_values('Wins Together', ascending=False).head(20)

        st.dataframe(
            synergy_df.style.background_gradient(subset=['Wins Together'], cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )

    # ── Tab 4: Patch Meta Shift ───────────────────────────────────
    with tab4:
        st.subheader("Patch Meta Shift")
        st.caption("How hero pick rates have changed across patches")

        patch_order_clean = [p for p in PATCH_ORDER if p in df['patch'].values]

        rows = []
        for patch_val in patch_order_clean:
            df_patch = df[df['patch'] == patch_val]
            if len(df_patch) < 10:
                continue
            patch_picks = pd.concat([df_patch[c] for c in pick_cols_raw]).dropna()
            total = len(df_patch) * 10
            counts = patch_picks.value_counts()
            for hero, count in counts.items():
                rows.append({
                    'patch': patch_val,
                    'hero': hero,
                    'pick_rate': round(count / total * 100, 2)
                })

        meta_df = pd.DataFrame(rows)

        top_heroes = all_picks.value_counts().head(20).index.tolist()
        selected_heroes = st.multiselect(
            "Select heroes to compare (max 8)",
            options=[h.title() for h in top_heroes],
            default=[h.title() for h in top_heroes[:5]],
            max_selections=8
        )

        if selected_heroes:
            selected_lower = [h.lower() for h in selected_heroes]
            chart_df = meta_df[meta_df['hero'].isin(selected_lower)].copy()
            chart_df['hero'] = chart_df['hero'].str.title()
            chart_pivot = chart_df.pivot(index='patch', columns='hero', values='pick_rate').fillna(0)
            chart_pivot = chart_pivot.reindex([p for p in patch_order_clean if p in chart_pivot.index])
            st.line_chart(chart_pivot)
            st.caption("Pick rate (%) per patch. Higher = more picked in that patch.")
        else:
            st.info("Select at least one hero to display the chart.")

    # ── Tab 5: Counter Picks ──────────────────────────────────────
    with tab5:
        st.subheader("Counter Pick Analysis")
        st.caption("Heroes that win more often when facing a specific enemy hero (min 20 games together)")

        counter_data = {}

        for _, row in df_filtered.iterrows():
            winner = row.get('winner')
            if str(winner) == '1':
                win_picks  = [row.get(f'team1pick{n}') for n in range(1,6)]
                lose_picks = [row.get(f'team2pick{n}') for n in range(1,6)]
            elif str(winner) == '2':
                win_picks  = [row.get(f'team2pick{n}') for n in range(1,6)]
                lose_picks = [row.get(f'team1pick{n}') for n in range(1,6)]
            else:
                continue

            win_picks  = [h for h in win_picks  if pd.notna(h)]
            lose_picks = [h for h in lose_picks if pd.notna(h)]

            for winner_hero in win_picks:
                for loser_hero in lose_picks:
                    key = (winner_hero, loser_hero)
                    if key not in counter_data:
                        counter_data[key] = {'wins': 0, 'total': 0}
                    counter_data[key]['wins']  += 1
                    counter_data[key]['total'] += 1
                    rev = (loser_hero, winner_hero)
                    if rev not in counter_data:
                        counter_data[rev] = {'wins': 0, 'total': 0}
                    counter_data[rev]['total'] += 1

        counter_rows = []
        for (hero, enemy), stats in counter_data.items():
            if stats['total'] >= 20:
                wr = round(stats['wins'] / stats['total'] * 100, 1)
                counter_rows.append({
                    'Hero': hero.title(),
                    'vs Enemy Hero': enemy.title(),
                    'Games': stats['total'],
                    'Win Rate %': wr
                })

        counter_df = pd.DataFrame(counter_rows)\
            .sort_values('Win Rate %', ascending=False)

        search_hero = st.selectbox(
            "Search counters for a specific hero",
            ["All"] + sorted(set(h.title() for col in pick_cols_raw for h in df_filtered[col].dropna().unique()))
        )

        if search_hero != "All":
            display_df = counter_df[counter_df['vs Enemy Hero'] == search_hero]\
                .sort_values('Win Rate %', ascending=False).head(15)
            st.caption(f"Heroes that win most against {search_hero}")
        else:
            display_df = counter_df.head(20)

        if search_hero != "All" and display_df.empty:
            if selected_patch == "All Patches":
                st.info(
                    f"{search_hero} has not faced any single enemy hero at least 20 times "
                    f"across the entire dataset, likely due to a low overall pick rate, "
                    f"so no reliable counter pick data is available for this hero."
                )
            else:
                st.info(
                    f"No hero has faced {search_hero} at least 20 times on patch {selected_patch}, "
                    f"so no reliable counter pick data is available. Try selecting 'All Patches' for a larger sample."
                )
        else:
            st.dataframe(
                display_df.style.background_gradient(subset=['Win Rate %'], cmap='RdYlGn'),
                use_container_width=True,
                hide_index=True
            )

    # ── Tab 6: Win Rate ───────────────────────────────────────────
    with tab6:
        st.subheader("Hero Win Rates")
        st.caption("Win rate of each hero when picked, across all filtered games")

        win_rate_data = {}

        for _, row in df_filtered.iterrows():
            winner = row.get('winner')
            if str(winner) == '1':
                win_picks  = [row.get(f'team1pick{n}') for n in range(1,6)]
                lose_picks = [row.get(f'team2pick{n}') for n in range(1,6)]
            elif str(winner) == '2':
                win_picks  = [row.get(f'team2pick{n}') for n in range(1,6)]
                lose_picks = [row.get(f'team1pick{n}') for n in range(1,6)]
            else:
                continue

            for h in win_picks:
                if pd.notna(h):
                    if h not in win_rate_data:
                        win_rate_data[h] = {'wins': 0, 'games': 0}
                    win_rate_data[h]['wins']  += 1
                    win_rate_data[h]['games'] += 1

            for h in lose_picks:
                if pd.notna(h):
                    if h not in win_rate_data:
                        win_rate_data[h] = {'wins': 0, 'games': 0}
                    win_rate_data[h]['games'] += 1

        wr_rows = []
        for hero, stats in win_rate_data.items():
            if stats['games'] >= 20:
                wr = round(stats['wins'] / stats['games'] * 100, 1)
                wr_rows.append({
                    'Hero': hero.title(),
                    'Games Played': stats['games'],
                    'Wins': stats['wins'],
                    'Win Rate %': wr
                })

        wr_df = pd.DataFrame(wr_rows).sort_values('Win Rate %', ascending=False)

        wr_col1, wr_col2 = st.columns(2)

        with wr_col1:
            st.markdown("**🏆 Highest Win Rate Heroes**")
            st.dataframe(
                wr_df.head(15).style.background_gradient(subset=['Win Rate %'], cmap='Greens'),
                use_container_width=True,
                hide_index=True
                    )

        with wr_col2:
            st.markdown("**📉 Lowest Win Rate Heroes**")
            st.dataframe(
                wr_df.tail(15).sort_values('Win Rate %').style.background_gradient(
                    subset=['Win Rate %'], cmap='Reds_r'),
                use_container_width=True,
                hide_index=True
                    )

        st.markdown("---")
        st.markdown("**🏟️ Win Rate by Tournament**")
        st.caption("Average win rate of heroes per tournament — shows how meta varies across regions")

        all_tournaments = sorted(df_filtered['tournament'].dropna().unique())
        selected_tourn = st.selectbox(
            "Select Tournament",
            ["All"] + list(all_tournaments),
            key="tourn_select_tab6"
        )

        if selected_tourn != "All":
            df_tourn = df_filtered[df_filtered['tournament'] == selected_tourn]
        else:
            df_tourn = df_filtered

        tourn_wr = {}
        for _, row in df_tourn.iterrows():
            winner = row.get('winner')
            if str(winner) == '1':
                wp = [row.get(f'team1pick{n}') for n in range(1,6)]
                lp = [row.get(f'team2pick{n}') for n in range(1,6)]
            elif str(winner) == '2':
                wp = [row.get(f'team2pick{n}') for n in range(1,6)]
                lp = [row.get(f'team1pick{n}') for n in range(1,6)]
            else:
                continue
            for h in wp:
                if pd.notna(h):
                    if h not in tourn_wr:
                        tourn_wr[h] = {'wins': 0, 'games': 0}
                    tourn_wr[h]['wins']  += 1
                    tourn_wr[h]['games'] += 1
            for h in lp:
                if pd.notna(h):
                    if h not in tourn_wr:
                        tourn_wr[h] = {'wins': 0, 'games': 0}
                    tourn_wr[h]['games'] += 1

        tourn_rows = [
            {'Hero': h.title(), 'Games': s['games'], 'Win Rate %': round(s['wins']/s['games']*100, 1)}
        for h, s in tourn_wr.items() if s['games'] >= 5
        ]

        if tourn_rows:
            tourn_df = pd.DataFrame(tourn_rows)

            sort_col1, sort_col2 = st.columns(2)
            with sort_col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Win Rate %", "Hero", "Games"],
                    key="tourn_sort_by"
                )
            with sort_col2:
                sort_order = st.selectbox(
                    "Order",
                    ["Descending", "Ascending"],
                    key="tourn_sort_order"
                )

            ascending = (sort_order == "Ascending")
            tourn_df = tourn_df.sort_values(sort_by, ascending=ascending)

            st.dataframe(
                tourn_df.head(20).style.background_gradient(subset=['Win Rate %'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Not enough data for this tournament filter.")