# Third Party Imports
import numpy as np
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_anomaly_score(anomaly_scores: np.ndarray) -> go.Figure:
    # Has to be numpy array
    if not isinstance(anomaly_scores, np.ndarray):
        raise ValueError("anomaly_scores must be a numpy array")
    # if not i Dimension array
    if anomaly_scores.ndim != 1:
        raise ValueError("anomaly_scores must be a 1D array")

    # The Figure
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            "<b>Anomaly Scores Histogram</b>",
            "<b>Anomaly Scores Boxplot</b>",
        ],
    )
    # Add a scatter plot
    fig.add_trace(
        go.Histogram(x=anomaly_scores, showlegend=False, name="Anomaly Scores"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Box(x=anomaly_scores, showlegend=False, name="Anomaly Scores"), row=2, col=1
    )
    # axis
    fig.update_xaxes(title_text="<b>Anomaly Scores</b>")
    fig.update_yaxes(title_text="<b>Count</b>", row=1, col=1)
    # Set Layout
    fig.update_layout(
        title=dict(text="<b>Anomaly Scores</b>", x=0.5, font=dict(size=30)),
        showlegend=False,
        template="plotly_dark",
        height=750,
    )
    # Show the figure
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> go.Figure:
    # Has to be numpy array
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred_proba, np.ndarray):
        raise ValueError("y_pred_proba must be a numpy array")
    # if not i Dimension array
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if y_pred_proba.ndim != 1:
        raise ValueError("y_pred_proba must be a 1D array")

    # Get the Threshold
    precision, recall, thresholds = precision_recall_curve(
        y_true=y_true, y_score=y_pred_proba, drop_intermediate=False
    )
    # Comute F1 scores
    f1_score = (
        2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    )

    # Plot the PR Curve
    fig = go.Figure()

    # Add the PR Curve
    fig.add_trace(
        go.Scatter(
            x=recall[:-1],
            y=precision[:-1],
            customdata=np.concatenate(
                [thresholds[:, np.newaxis], f1_score[:-1, np.newaxis]], axis=1
            ),
            mode="lines",
            name="Precision-Recall Curve",
            hovertemplate=(
                "Precision: %{y:.2f}<br>"
                "Recall: %{x:.2f}"
                "<br>F1 Score: %{customdata[1]:.2f}<br>"
                "Threshold: %{customdata[0]:.2f}"
            ),
        )
    )
    # The Id with max F1 score
    idx = f1_score[:-1].argmax()
    fig.add_trace(
        go.Scatter(
            x=[recall[idx]],
            y=[precision[idx]],
            customdata=np.array([[thresholds[idx], f1_score[idx]]]),
            name="Best Threshold",
            hovertemplate=(
                "Precision: %{y:.2f}<br>"
                "Recall: %{x:.2f}"
                "<br>F1 Score: %{customdata[1]:.2f}<br>"
                "Threshold: %{customdata[0]:.2f}"
            ),
        )
    )

    # Add the Title
    fig.update_layout(
        title=dict(text="Precision-Recall Curve", x=0.5, font=dict(size=35)),
        xaxis_title="<b>Recall</b>",
        yaxis_title="<b>Precision</b>",
        template="plotly_dark",
        width=700,
        showlegend=False,
    )

    # Show the Plot & Best Threshold
    return fig, float(thresholds[idx])


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> go.Figure:
    # Has to be numpy array
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true must be a numpy array")
    if not isinstance(y_pred_proba, np.ndarray):
        raise ValueError("y_pred_proba must be a numpy array")
    # if not i Dimension array
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if y_pred_proba.ndim != 1:
        raise ValueError("y_pred_proba must be a 1D array")

    # ROC Curves
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_proba)

    # Plot the PR Curve
    fig = go.Figure()

    # Add the PR Curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            customdata=thresholds,
            mode="lines",
            name="ROC Curve",
            hovertemplate=(
                "TPR: %{y:.2f}<br>FPR: %{x:.2f}<br>Threshold: %{customdata:.2f}"
            ),
        )
    )

    # Add the Title
    fig.update_layout(
        title=dict(text="ROC Curve", x=0.5, font=dict(size=35)),
        xaxis_title="<b>FPR</b>",
        yaxis_title="<b>TPR</b>",
        template="plotly_dark",
        width=700,
        showlegend=False,
    )

    # ShowPlot
    return fig
