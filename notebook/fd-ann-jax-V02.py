import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full", app_title="Fraud Detection With ANN")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **JAX Based ANN for Fraud Detection**
    """)
    return


@app.cell
def _():
    # Standard Imports
    from pprint import pprint
    from functools import partial
    from itertools import pairwise
    from typing import Self, Sequence, NamedTuple, Callable, Any

    # Third Party imports
    import jax
    import optax
    import marimo as mo
    import pandas as pd
    import polars as pl
    import jax.numpy as jnp
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        roc_auc_score,
        precision_recall_curve,
        average_precision_score,
    )
    from tqdm import tqdm

    # Internal Imports
    from fraud_detection.config import load_config, Config

    return (
        Any,
        Callable,
        Config,
        NamedTuple,
        Self,
        Sequence,
        average_precision_score,
        go,
        jax,
        jnp,
        load_config,
        mo,
        optax,
        pairwise,
        partial,
        pd,
        pl,
        pprint,
        precision_recall_curve,
        roc_auc_score,
        tqdm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Config & Validation**
    """)
    return


@app.cell
def _(load_config, pprint):
    # Load the config
    config, cfg_raw = load_config("./config/base.yaml")

    # Display config
    pprint(cfg_raw)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Load Data**
    """)
    return


@app.cell
def _(Config, jax, jnp, pd, pl, train_test_split):
    def get_data(
        config: Config,
    ) -> tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
        list[str],
    ]:
        # Read data
        df = pl.read_csv(
            source=config.data.input_path,
            ignore_errors=config.read_csv.ignore_errors,
            infer_schema_length=config.read_csv.infer_schema_length,
        )

        # Split data into X and y
        x = df.select(pl.exclude("Class")).to_pandas()
        y = df.select("Class").to_series().to_pandas()

        # Set type of the splits
        X_train: pd.DataFrame
        y_train: pd.Series
        X_test: pd.DataFrame
        y_test: pd.Series

        # Test Data Split
        X_train, X_test, y_train, y_test = train_test_split(
            x,
            y,
            stratify=y,
            train_size=config.split.test_split.train_size,
            random_state=config.split.test_split.random_state,
        )

        # Valid Data Split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            stratify=y_train,
            train_size=config.split.valid_split.train_size,
            random_state=config.split.valid_split.random_state,
        )

        # # The Target Mapping
        # target_mapping = {0: "Legitimate", 1: "Fraudulent"}

        # Convert DataFrame to Jax arrays
        X_train_jnp = jnp.array(X_train.values, dtype=jnp.float32)
        y_train_jnp = jnp.array(y_train.values, dtype=jnp.int32).reshape(-1, 1)
        X_valid_jnp = jnp.array(X_valid.values, dtype=jnp.float32)
        y_valid_jnp = jnp.array(y_valid.values, dtype=jnp.int32).reshape(-1, 1)
        X_test_jnp = jnp.array(X_test.values, dtype=jnp.float32)
        y_test_jnp = jnp.array(y_test.values, dtype=jnp.int32).reshape(-1, 1)

        # Feature names
        feature_names = [str(_) for _ in X_train.columns]

        return (
            (X_train_jnp, y_train_jnp),
            (X_valid_jnp, y_valid_jnp),
            (X_test_jnp, y_test_jnp),
            feature_names,
        )

    return (get_data,)


@app.cell
def _(config, get_data):
    # Read and Process data
    (
        (X_train_jnp, y_train_jnp),
        (X_valid_jnp, y_valid_jnp),
        (X_test_jnp, y_test_jnp),
        feature_names,
    ) = get_data(config=config)
    return X_test_jnp, X_train_jnp, X_valid_jnp, y_train_jnp, y_valid_jnp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *Data Processing*
    """)
    return


@app.cell
def _(Self, jax, jnp):
    class ShapeError(Exception):
        def __init__(self, message):
            super().__init__(message)
            self.message = message


    class NotFitError(Exception):
        def __init__(self: Self, message) -> None:
            super().__init__(message)
            self.message = message


    class StandardScaler:
        def __init__(self: Self) -> None:
            self.feat_mean: jax.Array | None = None
            self.feat_std: jax.Array | None = None

        def fit(self: Self, X: jax.Array) -> None:
            # Raise Shape Error if the array is not correct
            if X.ndim != 2:
                raise ShapeError(
                    "X input in fit method of Standard Scaler has to have dimension 2."
                )

            # Calulate the mean and std
            self.feat_mean = jnp.mean(X, axis=0)
            self.feat_std = jnp.std(X, axis=0)

        def transform(
            self: Self, X: jax.Array, epsilon: float = 1e-9
        ) -> jax.Array:
            # If the model was not fir before
            if (self.feat_mean is None) or (self.feat_std is None):
                raise NotFitError(
                    "The Standard Scaler need first fit to be run before transform"
                )

            return (X - self.feat_mean) / jnp.maximum(self.feat_std, epsilon)

        def fit_transform(
            self: Self, X: jax.Array, epsilon: float = 1e-9
        ) -> jax.Array:
            # Calulate the mean and std
            self.fit(X)
            return self.transform(X, epsilon)

    return (StandardScaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *Model & losses*
    """)
    return


@app.cell
def _(NamedTuple, Sequence, jax, jnp, pairwise, partial):
    class Weights(NamedTuple):
        w: jax.Array
        b: jax.Array


    def build_ann(sizes: Sequence[int], key_seed: int = 42):
        # Compute matric IO Sizes
        mat_io = list(pairwise(sizes))

        # Get the master Key
        master_key = jax.random.key(seed=key_seed)
        key_forward = master_key

        # The weights
        weights = []
        # iterate through te layers
        for mat_dim in mat_io:
            # split the keys
            mat_key, bias_key, key_forward = jax.random.split(key_forward, 3)
            # declare weights
            weights.append(
                Weights(
                    jax.random.normal(key=mat_key, shape=mat_dim)
                    * jnp.sqrt((2 / mat_dim[0])),
                    jnp.zeros(shape=(mat_dim[1],)),
                )
            )

        # Return wights
        return weights


    @partial(jax.jit, static_argnames=["dropout"])
    def predict_logits(
        weights: list[Weights],
        X: jax.Array,
        dropout: bool = False,
        dorpout_rate: float = 0.1,
        dropout_key: jax.random.PRNGKey = jax.random.key(seed=42),
    ):
        # The output res init
        output = X

        # loop the weights
        for w, b in weights[:-1]:
            # Compute Output for each layer
            output = jnp.maximum(jnp.einsum("io,ni->no", w, output) + b, 0)

            if dropout:
                # Split the keys
                dropout_key, _ = jax.random.split(dropout_key)
                # Calculate the droupout mask
                mask = jax.random.bernoulli(
                    key=dropout_key, p=1 - dorpout_rate, shape=output.shape
                ) * (1 / (1 - dorpout_rate))
                # Mask the output
                output = jnp.einsum("ab,ab->ab", output, mask)
        # Last Layer
        w, b = weights[-1]
        # Final Logits
        return jnp.einsum("io,ni->no", w, output) + b


    @jax.jit()
    def weighted_bce(
        logits: jax.Array,
        y: jax.Array,
        pos_weight: float = 1.0,
    ) -> float:
        # Conver the
        y = y.astype(logits.dtype)
        # Flatten the arrays
        c = 1 + y * (pos_weight - 1)
        # The weighted BCE loss
        loss_points = -(
            (pos_weight * y * logits)
            - (c * (jnp.maximum(logits, 0) + jax.nn.softplus(-jnp.abs(logits))))
        )
        # Return mean
        return jnp.mean(loss_points)


    @partial(jax.jit, static_argnames=["dropout"])
    def forward_pass(
        weights: list[Weights],
        X: jax.Array,
        y: jax.Array,
        pos_weight: float = 1.0,
        dropout: bool = False,
        dorpout_rate: float = 0.1,
        dropout_key: jax.random.PRNGKey = jax.random.key(seed=42),
    ):
        # Logits
        logits = predict_logits(weights, X, dropout, dorpout_rate, dropout_key)
        # BCE Loss function
        loss = weighted_bce(logits=logits, y=y, pos_weight=pos_weight)
        # Return
        return loss


    @partial(jax.jit, static_argnames=["dropout"])
    def forward_grad_loss(
        weights: list[Weights],
        X: jax.Array,
        y: jax.Array,
        pos_weight: float = 1.0,
        dropout: bool = False,
        dorpout_rate: float = 0.1,
        dropout_key: jax.random.PRNGKey = jax.random.key(seed=42),
    ):
        return jax.value_and_grad(forward_pass)(
            weights, X, y, pos_weight, dropout, dorpout_rate, dropout_key
        )

    return Weights, build_ann, forward_grad_loss, forward_pass, predict_logits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## *Training Functions*
    """)
    return


@app.cell
def _(Any, Callable, NamedTuple, jax, jnp):
    class BatchState(NamedTuple):
        idx: int
        indices: jax.Array
        key: jax.random.PRNGKey


    def get_batchstate_fuc(
        X: jax.Array, y: jax.Array, batch_size: int = 512, seed: int = 1237
    ) -> tuple[BatchState, Callable[Any, Any]]:
        # Init the Bach array
        batch_array: jax.Array = jnp.arange(0, batch_size)
        # init the BatchState
        batch_state = BatchState(
            idx=0,
            indices=jnp.arange(X.shape[0]),
            key=jax.random.key(seed),
        )

        @jax.jit()
        def _update_batch(batch_state: BatchState) -> BatchState:
            return BatchState(
                idx=batch_state.idx + batch_size,
                indices=batch_state.indices,
                key=batch_state.key,
            )

        @jax.jit()
        def _reset_batch(batch_state: BatchState) -> BatchState:
            # Update key
            key, _ = jax.random.split(batch_state.key)
            # Shuffle the indeices
            indices = jax.random.permutation(key=key, x=batch_state.indices)
            # Return Batch
            return BatchState(
                idx=0,
                indices=indices,
                key=key,
            )

        @jax.jit()
        def _get_batch(batch_state: BatchState, X: jax.Array, y: jax.Array):
            # The start and stop index
            start_idx = batch_state.idx
            end_idx = batch_state.idx + batch_size

            # Data Sixe
            data_size = X.shape[0]

            # The decision
            decision_reset = end_idx >= data_size

            # selected indices
            slected_indices = jnp.take(
                batch_state.indices,
                (start_idx + batch_array) % data_size,
                mode="clip",
                axis=0,
            )

            # get sample
            X_batch = jnp.take(X, slected_indices, mode="clip", axis=0)
            y_batch = jnp.take(y, slected_indices, mode="clip", axis=0)

            # Update & Reset batch state
            batch_state = jax.lax.cond(
                decision_reset, _reset_batch, _update_batch, batch_state
            )

            # return data
            return batch_state, X_batch, y_batch

        return batch_state, _get_batch

    return BatchState, get_batchstate_fuc


@app.cell
def _(
    Any,
    BatchState,
    NamedTuple,
    Weights,
    forward_grad_loss,
    get_batchstate_fuc,
    jax,
    optax,
):
    # Train State
    class TrainState(NamedTuple):
        weights: Weights
        opt_state: Any
        train_batchstate: BatchState
        valid_batchstate: BatchState
        dropout_key: jax.random.PRNGKey


    class LossState(NamedTuple):
        train_loss: jax.Array


    def factory_train_step(
        optimizer,
        X_train: jax.Array,
        y_train: jax.Array,
        X_valid: jax.Array,
        y_valid: jax.Array,
        batch_size: int = 32,
        seed: int = 42,
    ):
        # The Batch init and funcs
        train_batchstate, func_get_train_batch = get_batchstate_fuc(
            X_train, y_train, batch_size, seed
        )
        valid_batchstate, func_get_valid_batch = get_batchstate_fuc(
            X_valid, y_valid, batch_size, seed
        )

        def train_step(
            train_state: TrainState,
            X_train: jax.Array,
            y_train: jax.Array,
            X_valid: jax.Array,
            y_valid: jax.Array,
            pos_weight: float = 1.0,
            dropout: bool = False,
            dorpout_rate: float = 0.1,
        ) -> tuple[TrainState, LossState]:
            # Extract state
            weights = train_state.weights
            opt_state = train_state.opt_state
            train_batchstate = train_state.train_batchstate
            valid_batchstate = train_state.valid_batchstate
            dropout_key = train_state.dropout_key

            # Split the key
            dropout_key, _ = jax.random.split(dropout_key)

            # Get Train & and batch state update
            train_batchstate, X_train_batch, y_train_batch = func_get_train_batch(
                train_batchstate, X_train, y_train
            )
            # Get Valid & Batch State
            valid_batchstate, X_valid_batch, y_valid_batch = func_get_valid_batch(
                valid_batchstate, X_valid, y_valid
            )

            # Loss and Gradinets
            train_loss, grads = forward_grad_loss(
                weights,
                X_train_batch,
                y_train_batch,
                pos_weight,
                dropout,
                dorpout_rate,
                dropout_key,
            )

            # Update states
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)

            # Return
            return TrainState(
                weights, opt_state, train_batchstate, valid_batchstate, dropout_key
            ), LossState(
                train_loss,
            )

        return train_batchstate, valid_batchstate, train_step

    return TrainState, factory_train_step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Training Model**
    """)
    return


@app.cell
def _(StandardScaler, X_test_jnp, X_train_jnp, X_valid_jnp, jnp, y_train_jnp):
    # Set Standard Transfrom
    ss = StandardScaler()

    # Transform data
    X_train_ss = ss.fit_transform(X_train_jnp)
    X_valid_ss = ss.transform(X_valid_jnp)
    X_test_ss = ss.transform(X_test_jnp)  # noqa: F841

    # positve weights
    pos_weight = (y_train_jnp.size - y_train_jnp.sum()) / y_train_jnp.sum()
    pos_weight = jnp.sqrt(pos_weight)
    print(pos_weight)
    print(pos_weight**2)
    return X_train_ss, X_valid_ss, pos_weight


@app.cell
def _(X_train_jnp, build_ann):
    #  Input controls
    input_features = X_train_jnp.shape[1]

    # Get Wrights
    # weights = build_ann([input_features, 1])  # Logentic Regression
    # weights = build_ann([input_features, 15, 3, 15, 1])  # The AE + CLF
    weights = build_ann([input_features, 20, 10, 1])  # DNN
    # weights = build_ann([input_features, 10, 1])  # DNN2
    return (weights,)


@app.cell
def _(optax, weights):
    # Iteration
    epoches = 100

    #  Input controls
    learning_rate = 1e-3
    batch_size = 1024
    batch_seed = 4545

    ## Drop out controls
    dropout: bool = True
    dorpout_rate: float = 0.4
    dropout_seed: int = 463

    # Set Up optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=optax.cosine_decay_schedule(
                learning_rate, decay_steps=600
            ),
            weight_decay=1e-2,
        ),
    )

    # Optimzer state
    opt_state = optimizer.init(weights)
    return (
        batch_seed,
        batch_size,
        dorpout_rate,
        dropout,
        dropout_seed,
        epoches,
        opt_state,
        optimizer,
    )


@app.cell
def _(
    X_train_ss,
    X_valid_ss,
    batch_seed,
    batch_size,
    dorpout_rate: float,
    dropout: bool,
    factory_train_step,
    jax,
    optimizer,
    pos_weight,
    y_train_jnp,
    y_valid_jnp,
):
    # Train state
    train_batchstate, valid_batchstate, train_step = factory_train_step(
        optimizer,
        X_train_ss,
        y_train_jnp,
        X_valid_ss,
        y_valid_jnp,
        batch_size,
        batch_seed,
    )

    # The carry over function for training
    train_step_scan = jax.jit(
        lambda train_state, _: train_step(
            train_state=train_state,
            X_train=X_train_ss,
            y_train=y_train_jnp,
            X_valid=X_valid_ss,
            y_valid=y_valid_jnp,
            pos_weight=pos_weight,
            dropout=dropout,
            dorpout_rate=dorpout_rate,
        )
    )
    return train_batchstate, train_step_scan, valid_batchstate


@app.cell
def _(
    TrainState,
    X_train_ss,
    X_valid_ss,
    batch_size,
    dropout_seed: int,
    epoches,
    forward_pass,
    jax,
    opt_state,
    pos_weight,
    tqdm,
    train_batchstate,
    train_step_scan,
    valid_batchstate,
    weights,
    y_valid_jnp,
):
    # Interation contol
    steps_per_epoch = X_train_ss.shape[0] // batch_size

    # Set Initial Batch state
    train_state = TrainState(
        weights,
        opt_state,
        train_batchstate,
        valid_batchstate,
        dropout_key=jax.random.key(seed=dropout_seed),
    )

    # History of Data Saves
    train_loss_list = []
    valid_loss_list = []

    # Train Iteration
    for _ in tqdm(range(epoches)):
        # Training data
        train_state, loss_hsitory = jax.lax.scan(
            train_step_scan, init=train_state, length=steps_per_epoch
        )
        # Get Validation predictions
        vlidation_loss = forward_pass(
            train_state.weights, X_valid_ss, y_valid_jnp, pos_weight
        )
        # Save history
        train_loss_list.append(loss_hsitory)
        valid_loss_list.append(vlidation_loss)
    return train_loss_list, train_state, valid_loss_list


@app.cell
def _(epoches, go, train_loss_list, valid_loss_list):
    # Show Figure
    fig = go.Figure()

    # Show plots
    fig.add_trace(
        go.Scatter(
            y=[_.train_loss.mean() for _ in train_loss_list], name="Train BCE Loss"
        )
    )
    fig.add_trace(go.Scatter(y=valid_loss_list, name="Valid BCE Loss"))

    # Layout shown
    fig.update_layout(
        title={
            "text": (
                "<b>BCE Loss for Train and Validation Sets. "
                f"Epochs {epoches}, "
                f"Min Valid Loss {min(valid_loss_list):.2f}</b>"
            ),
            "x": 0.5,
        },
    )

    # Display figure
    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # **Train Data Probabality Caliberation**
    """)
    return


@app.cell
def _(X_valid_ss, jax, predict_logits, train_state):
    # Test data probabality
    y_valid_proba = jax.nn.sigmoid(predict_logits(train_state.weights, X_valid_ss))
    return (y_valid_proba,)


@app.cell
def _(average_precision_score, roc_auc_score, y_valid_jnp, y_valid_proba):
    print(f"AP Score = {average_precision_score(y_valid_jnp, y_valid_proba):.3f}")
    print(f"ROC-AUC  = {roc_auc_score(y_valid_jnp, y_valid_proba):.3f}")
    return


@app.cell
def _(precision_recall_curve, y_valid_jnp, y_valid_proba):
    # Precision Recall and Threshold
    precision, recall, pr_threshold = precision_recall_curve(
        y_valid_jnp, y_valid_proba
    )

    # F1 Scores
    f1_scores = (2 * precision * recall / (precision + recall))[:-1]

    # Best Index
    best_idx = f1_scores.argmax()
    return best_idx, f1_scores, pr_threshold, precision, recall


@app.cell
def _(
    average_precision_score,
    best_idx,
    f1_scores,
    go,
    pr_threshold,
    precision,
    recall,
    y_valid_jnp,
    y_valid_proba,
):
    # Show Figure
    fig2 = go.Figure()

    # PR Curve
    fig2.add_trace(
        go.Scatter(
            x=recall[:-1],
            y=precision[:-1],
            mode="lines",
            name="PR Curve",
            line=dict(color="#00BFFF", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 191, 255, 0.08)",
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=[recall[best_idx]],
            y=[precision[best_idx]],
            mode="markers+text",
            name=f"Best F1: {f1_scores.max():.3f}",
            marker=dict(color="#FF4500", size=12, symbol="circle"),
            text=[f"  Threshold: {pr_threshold[best_idx]:.3f}"],
            textposition="middle right",
            textfont=dict(color="white", size=11),
        )
    )
    # Layout
    fig2.update_layout(
        template="plotly_dark",
        title=dict(
            text=(
                "<b>Precision-Recall Curve — Test Set</b><br>"
                f"<sup>PR-AUC: {average_precision_score(y_valid_jnp, y_valid_proba):.4f}</sup>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Recall",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[0, 1],
        ),
        yaxis=dict(
            title="Precision",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[0, 1.05],
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        width=900,
        height=550,
    )

    fig2.show()
    return


@app.cell
def _(jax, jnp):
    def NDCG_at_k(correct_vec: jax.Array, R: int, k: int):
        # Postion vector
        pos_vec = jnp.arange(1, k + 1)
        # Ideal Postion Vector
        ideal_pos_vec = jnp.arange(1, jnp.minimum(k, R) + 1)
        # Discounted Cumulative Gains
        DCG = (correct_vec / jnp.log2(pos_vec + 1)).sum()

        # Ideal Discounted Cumulative Gains
        IDCG = (1 / jnp.log2(ideal_pos_vec + 1)).sum()

        # Normalized Discounted Cumulative Gains
        NDCG = DCG / IDCG

        # return  Normalized Dicounted Cumulative Gains
        return NDCG


    def top_k_metrics(
        y_true: jax.Array, y_pred: jax.Array, threshold: float = 0.444, k: int = 10
    ):
        # Ordering index
        ids = jnp.argsort(y_pred.ravel(), descending=True)

        # The sorted array
        y_pred_sort = y_pred.ravel()[ids]
        y_true_sort = y_true.ravel()[ids]
        # The top k pred

        y_pred_topk = (y_pred_sort >= threshold).astype(jnp.int32)[:k]
        y_true_topk = y_true_sort[:k]

        # Inersections
        correct_ans_vec = ((y_pred_topk + y_true_topk) == 2).astype(jnp.int32)
        correct_ans = correct_ans_vec.sum()

        # Base Rate
        R = y_true.sum()
        base_rate = R / y_true.size

        # base Rate
        convergence = correct_ans / jnp.maximum(y_true_topk.sum(), 1e-10)
        recall = correct_ans / R
        precision = correct_ans / k

        # Lift @ K
        lift = precision / base_rate

        # Average pricsion @ Top-K
        moving_pr = jnp.cumsum(correct_ans_vec) / jnp.arange(
            1, k + 1, dtype=jnp.float32
        )
        ap = (moving_pr * correct_ans_vec).sum() / jnp.minimum(k, R)

        # Normalized Discounted Cumulative gains
        NDCG = NDCG_at_k(correct_vec=correct_ans_vec, k=k, R=R)

        # Return
        return {
            "Recall": recall,
            "Precision": precision,
            "Lift": lift,
            "AP": ap,
            "Convergence": convergence,
            "NDCG": NDCG,
            "base-rate": base_rate,
            "R": R,
        }

    return (top_k_metrics,)


@app.cell
def _(pd, top_k_metrics, y_valid_jnp, y_valid_proba):
    pd.DataFrame(
        [
            {
                "k": k,
                **top_k_metrics(y_valid_jnp, y_valid_proba, threshold=0.409, k=k),
            }
            for k in [5, 10, 20, 50, 100, 200, 500, 1000]
        ]
    )
    return


if __name__ == "__main__":
    app.run()
