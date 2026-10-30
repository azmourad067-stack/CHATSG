"""Résultats de la validation glissante (walk-forward)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from horse_intelligence.frontend.components.cards import EMERALD, metric_card, notice
from horse_intelligence.frontend.state import get_context
from horse_intelligence.services.training import REFERENCE_MODEL, latest_report

LABELS = {"uniform": "Uniforme", "market": "Marché brut", REFERENCE_MODEL: "Marché calibré",
          "form_logit": "Logit conditionnel (forme)", "lgbm_form": "LightGBM (forme)",
          "lgbm_market": "LightGBM + marché"}


def render() -> None:
    st.title("Backtesting")
    rep = latest_report(get_context().settings.reports_dir)
    if rep is None:
        notice("Aucune évaluation disponible. Lancez : python scripts/train_models.py")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Période", f"{rep['period'][0]} → {rep['period'][1]}")
    with c2:
        metric_card("Courses", str(rep["n_races"]), f"{rep['n_races_with_market']} avec cotes")
    with c3:
        metric_card("Plis walk-forward", str(len(rep["folds"])))
    with c4:
        audit = rep["source_aggregates_audit"]
        metric_card("Audit des fuites", "réussi" if audit["passed"] else "échec",
                    f"{audit['n_winners']} gagnants contrôlés")
    for w in rep.get("warnings", []):
        st.warning(w)

    st.subheader("Qualité probabiliste (critère principal)")
    m = pd.DataFrame(rep["metrics_market_races"]).T
    m.index = [LABELS.get(i, i) for i in m.index]
    st.dataframe(m[["n_races", "log_loss", "brier", "ece", "top1", "winner_in_top3", "auc"]]
                 .rename(columns={"n_races": "Courses", "log_loss": "Log-loss", "brier": "Brier",
                                  "ece": "ECE", "top1": "Top-1", "winner_in_top3": "Top-3",
                                  "auc": "AUC"}).round(4), width="stretch")
    st.caption("Même échantillon de courses (avec cotes de référence) pour tous les modèles. "
               "Log-loss et Brier : plus bas = meilleur.")

    st.subheader("Comparaison au marché calibré")
    d = pd.DataFrame(rep["vs_reference"]).T
    d.index = [LABELS.get(i, i) for i in d.index]
    d["Verdict"] = np.where(d["ci_high"].astype(float) < 0, "meilleur (IC < 0)",
                            np.where(d["ci_low"].astype(float) > 0, "moins bon (IC > 0)",
                                     "non significatif"))
    st.dataframe(d.rename(columns={"mean_diff": "Δ log-loss", "ci_low": "IC bas",
                                   "ci_high": "IC haut", "n_races": "Courses"}).round(4),
                 width="stretch")

    st.subheader("Calibration")
    options = list(rep["calibration"])
    default = options.index("lgbm_market") if "lgbm_market" in options else 0
    choice = st.selectbox("Modèle", options, index=default,
                          format_func=lambda k: LABELS.get(k, k))
    cal = pd.DataFrame(rep["calibration"][choice])
    fig = go.Figure()
    fig.add_scatter(x=[0, cal["p_mean"].max()], y=[0, cal["p_mean"].max()], mode="lines",
                    line={"dash": "dot", "color": "#8B98A5"}, name="Calibration parfaite")
    fig.add_scatter(x=cal["p_mean"], y=cal["y_rate"], mode="lines+markers",
                    line={"color": EMERALD}, name="Observé")
    fig.update_layout(height=320, template="plotly_dark", xaxis_title="Probabilité prédite",
                      yaxis_title="Fréquence de victoire", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Variables les plus utilisées (LightGBM forme)")
    st.bar_chart(pd.Series(rep["feature_importance"]).sort_values())
    st.caption("Importance = gain cumulé dans les arbres : indique l'usage d'une variable, "
               "pas un lien de causalité.")
