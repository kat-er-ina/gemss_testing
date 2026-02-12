import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import os

    # Add the parent directory to sys.path so 'gemss' can be imported
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    return current_dir, os


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import io
    from plotly import io as pio

    # GEMSS feature selector
    from gemss.feature_selection.inference import BayesianFeatureSelector

    # Visualizations
    from gemss.utils.visualizations import (
        show_label_histogram,
        show_final_alphas,
        show_features_in_components,
        get_algorithm_progress_plots,
        get_final_alphas_plot,
        get_label_histogram_plot,
        get_label_piechart,
        get_features_in_components_plot,
    )

    # Postprocessing
    from gemss.utils.utils import (
        get_solution_summary_df,
        save_feature_lists_json,
        save_feature_lists_txt,
        save_selector_history_json,
        save_constants_json,
    )
    from gemss.postprocessing.result_postprocessing import (
        recover_solutions,
        get_features_from_solutions,
        get_unique_features,
    )
    from gemss.postprocessing.simple_regressions import (
        detect_task,
        solve_any_regression,
        show_regression_metrics,
    )
    from gemss.postprocessing.tabpfn_evaluation import (
        tabpfn_evaluate,
    )
    from gemss.data_handling.data_processing import (
        preprocess_features,
        get_df_from_X,
    )

    # Use default renderer or 'iframe' if plots don't show
    # pio.renderers.default = "notebook_connected"
    return (
        BayesianFeatureSelector,
        detect_task,
        get_algorithm_progress_plots,
        get_df_from_X,
        get_features_from_solutions,
        get_features_in_components_plot,
        get_final_alphas_plot,
        get_label_histogram_plot,
        get_label_piechart,
        get_solution_summary_df,
        get_unique_features,
        io,
        mo,
        np,
        pd,
        preprocess_features,
        recover_solutions,
        save_constants_json,
        save_feature_lists_json,
        save_feature_lists_txt,
        save_selector_history_json,
        solve_any_regression,
        tabpfn_evaluate,
    )


@app.cell
def _(current_dir, mo, os):
    logo_path = os.path.join(current_dir, "datamole_logo_wide.jpg")

    # Read and display logo
    logo_link = mo.Html(
        f"""
        <div style="text-align: right;">
            <a href="https://www.datamole.ai/" target="_blank">
                {mo.image(src=logo_path, width=1000, alt="Datamole").text}
            </a>
        </div>
    """
    )

    mo.image(src=logo_path, width=1000, alt="Datamole")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 💎 **GEMSS Explorer** [non-commercial]

    This app helps you discover **multiple distinct feature sets** that explain your data using GEMSS: Gaussian Ensemble for Multiple Sparse Solutions.
    """
    )
    return


@app.cell
def _(mo):
    # More details
    intro_help = mo.accordion(
        {
            " 📖 Read more": mo.md(
                """
                ### What you will get

                Instead of finding just one "best" set of features, GEMSS discovers **several most likely feature combinations** that predict your target variable comparably well. This is valuable when:

                - You have precious few samples and many more features (common e.g. in life sciences).
                - Multiple underlying mechanisms might explain your data.
                - You are striving for an interpretable model.
                - You want to engineer a multitude of nonlinear and combined features from your original set for exploratory purposes.
                - Your features are correlated.
                - When there is domain knowledge to be mined (a human in the loop).

                **Example:** Instead of "use features A, B, C", you might discover three solutions: {A, B, C}, {D, E, F}, {A, E, G} — each explaining the data through a different mechanism.

                ### General workflow overview

                **I. Data loading** - Upload and configure your dataset. <br>
                **II. Algorithm setup** - Configure hyperparameters of GEMSS feature selector. <br>
                **III. Feature selection** - Run Bayesian inference to discover multiple components that describe your data. <br>
                **IV. Solution recovery** - Extract one sparse solution from each component, obtaining an ensemble of feature sets. <br>
                **V. Model evaluation** - Validate each solution by a simple linear/logistic regression model and then a full predictive model.

                Each step builds on the previous one, so please follow the workflow in order.

                Model evaluation requires agreement with the [license of TabPFN](https://huggingface.co/Prior-Labs/tabpfn_2_5#licensing) modeling tool.
                """
            )
        }
    )

    mo.vstack(
        [
            intro_help,
            mo.md("<br>"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **1. Set up input and output**
    """
    )
    return


@app.cell
def _(mo):
    # More details
    data_help = mo.accordion(
        {
            " 📖 Guide": mo.md(
                """
                ### The input data
                - must be already cleaned and preprocessed,
                - should have features in columns and samples in rows,
                - must contain an index column,
                - must contain a target/label column: binary classification and regression are supported,
                - can contain missing values,
                - only numeric features are supported.

                ### Data scaling

                It is highly recommended to scale your data, unless they already have been.
                - Use the *minmax* scaling to simply squish all features' values to [0, 1] range.
                - Use the *standard* scaling when you assume the samples are normally distributed. Each feature will then have mean 0 and standard deviation 1.

                ### The outputs

                When saving is enabled, the following files will be created in the specified directory (names can be customized):
                - `search_history_results.json` - JSON file containing the history of the feature selection process (backup for possible alternative postprocessing)
                - `search_setup.json` - JSON file containing the configuration of the feature selection process (for experiment tracking)
                - `feature_lists.txt` - human-readable text file containing the discovered feature sets
                - `feature_lists.json` - same as above, but in JSON format for easier programmatical handling in further modeling
                """
            )
        }
    )

    mo.vstack(
        [
            data_help,
        ]
    )
    return


@app.cell
def _(current_dir, mo):
    # Save configuration
    checkbox_save_results = mo.ui.checkbox(
        value=True,
        label="Save results",
    )
    save_dir_input = mo.ui.text(
        value=f"{current_dir}\\results",
        label="Parent directory for saving this experiment",
    )

    save_experiment_id = mo.ui.number(1, 1000, value=1, step=1, label="Experiment ID")
    save_history_name = mo.ui.text(
        value="search_history_results",
        label="History filename (no extension)",
    )
    save_setup_name = mo.ui.text(
        value="search_setup", label="Setup filename (no extension)"
    )
    save_features_name = mo.ui.text(
        value="all_candidate_solutions", label="Features filename (no extension)"
    )

    mo.vstack(
        [
            mo.md("### 1.1 Output configuration"),
            checkbox_save_results,
        ]
    )
    return (
        checkbox_save_results,
        save_dir_input,
        save_experiment_id,
        save_features_name,
        save_history_name,
        save_setup_name,
    )


@app.cell
def _(
    checkbox_save_results,
    mo,
    save_dir_input,
    save_experiment_id,
    save_features_name,
    save_history_name,
    save_setup_name,
):
    # configure saving options, if saving is enabled
    save_results = checkbox_save_results.value
    if save_results:
        _display = mo.vstack(
            [
                mo.md(
                    f"*Output files will be saved in: {save_dir_input.value}/experiment_{save_experiment_id.value}*/"
                ),
                mo.accordion(
                    {
                        " Edit save location": mo.vstack(
                            [
                                save_dir_input,
                                save_experiment_id,
                            ]
                        )
                    }
                ),
                mo.accordion(
                    {
                        "Edit file names": mo.vstack(
                            [
                                save_history_name,
                                save_setup_name,
                                save_features_name,
                            ]
                        )
                    }
                ),
            ]
        )
    else:
        _display = None

    _display
    return (save_results,)


@app.cell
def _(mo):
    # UI Components for Data Loading
    file_uploader = mo.ui.file(
        kind="button", label="Upload CSV dataset", filetypes=[".csv"]
    )

    mo.vstack(
        [
            mo.md("### 1.2 Input data"),
            file_uploader,
            mo.md("<br>"),
        ]
    )
    return (file_uploader,)


@app.cell
def _(file_uploader, io, mo, pd):
    # Logic to load data from uploader
    if file_uploader.value:
        # Load CSV from the uploaded bytes
        _content = file_uploader.value[0].contents
        df_raw = pd.read_csv(io.BytesIO(_content))

        # Column selectors based on the uploaded file
        index_col_selector = mo.ui.dropdown(
            options=list(df_raw.columns),
            label="Index column",
            value=df_raw.columns[0] if not df_raw.empty else None,
        )
        label_col_selector = mo.ui.dropdown(
            options=list(df_raw.columns),
            label="Target/label column",
            value=df_raw.columns[-1] if not df_raw.empty else None,
        )
        scaling_selector = mo.ui.dropdown(
            options=["standard", "minmax", None],
            label="Scaling to use",
            value="minmax",
        )
        allowed_missing_percentage_selector = mo.ui.number(
            0,
            50,
            value=10,
            step=5,
            label="Missing data allowed in a feature [%]",
        )

        data_setup_ui = mo.vstack(
            [
                mo.md("<br>"),
                mo.md(
                    f"✅ **Data loaded:** `{file_uploader.value[0].name}` ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols)"
                ),
                mo.vstack([index_col_selector, label_col_selector, scaling_selector]),
            ]
        )
    else:
        df_raw = None
        index_col_selector = None
        label_col_selector = None
        scaling_selector = None
        data_setup_ui = None

    (
        mo.vstack(
            [
                mo.md("**Your loaded dataset:**"),
                mo.ui.table(df_raw),
                mo.md("<br>"),
                data_setup_ui,
            ]
        )
        if df_raw is not None
        else mo.vstack(
            [
                mo.md("---"),
            ]
        )
    )
    return (
        allowed_missing_percentage_selector,
        df_raw,
        index_col_selector,
        label_col_selector,
        scaling_selector,
    )


@app.cell
def _(
    allowed_missing_percentage_selector,
    df_raw,
    get_df_from_X,
    get_label_histogram_plot,
    get_label_piechart,
    index_col_selector,
    label_col_selector,
    mo,
    np,
    pd,
    preprocess_features,
    scaling_selector,
):
    # Stop if data not loaded
    mo.stop(df_raw is None, mo.md("*Please upload your dataset to proceed.*<br><hr>"))

    # Data Preprocessing
    try:
        _df_proc = df_raw.copy()
        if index_col_selector.value:
            _df_proc.set_index(index_col_selector.value, inplace=True)

        _response = _df_proc[label_col_selector.value]
        # Preprocess
        X, y, feature_map = preprocess_features(
            _df_proc,
            _response,
            dropna="response",
            allowed_missing_percentage=allowed_missing_percentage_selector.value,
            drop_non_numeric_features=True,
            apply_scaling=scaling_selector.value,
            verbose=False,
        )
        overall_nan_ratio = np.isnan(X).sum() / (X.shape[0] * X.shape[1])
        df_processed = get_df_from_X(X, feature_map)

    except Exception as e:
        mo.stop(True, mo.md(f"**Error processing data:** {str(e)}"))

    n_samples = df_processed.shape[0]
    n_features = df_processed.shape[1]
    _n_response_values = pd.Series(y).nunique()

    min_samples_allowed = 10
    min_features_allowed = 10
    too_few_samples = n_samples < min_samples_allowed
    too_few_features = n_features < min_features_allowed

    mo.stop(
        too_few_samples,
        mo.md(
            f"**Error: too few samples available:** {n_samples} < {min_samples_allowed}. Fix your dataset before proceeding."
        ),
    )
    mo.stop(
        too_few_features,
        mo.md(
            f"**Error: too few features available:** {n_features} < {min_features_allowed}. Fix your dataset before proceeding."
        ),
    )

    mo.vstack(
        [
            mo.md(
                f"""
                ✅ **Data preprocessed:**
                - no. samples: {n_samples}
                - no. features: {n_features}
                - no. unique response values: {_n_response_values}
                - missing data: {overall_nan_ratio}%
                """
            ),
            # show label distribution either as a pie chart or a histogram, depending on the number of unique values
            (
                get_label_piechart(y)
                if _n_response_values < 5
                else get_label_histogram_plot(y)
            ),
            mo.md("---"),
            mo.md("<br>"),
        ]
    )
    return X, df_processed, feature_map, n_features, y


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **2. The feature selection algorithm**

    Configure parameters of the GEMSS feature selection algorithm.
    """
    )
    return


@app.cell
def _(df_processed, mo):
    # UI for Algorithm Configuration
    mo.stop(df_processed is None, mo.md("*Please upload data first.*"))

    # Basic Settings
    n_candidates = mo.ui.number(
        1, 20, value=8, step=1, label="Number of components (candidate solutions)"
    )
    sparsity_est = mo.ui.number(
        1,
        15,
        value=4,
        step=1,
        label="Desired sparsity (no. features per component)",
    )

    # Advanced Settings
    adv_iter = mo.ui.number(500, 20000, value=3500, step=250, label="Iterations")
    adv_lr = mo.ui.number(0.0000, 0.1, value=0.002, step=0.0001, label="Learning rate")
    adv_batch = mo.ui.number(
        8,
        256,
        value=16,
        step=4,
        label="Batch size (no. samples in a minibatch)",
    )
    adv_jaccard = mo.ui.checkbox(
        value=True, label="Enforce diversity (penalize Jaccard similarity)"
    )
    adv_lambda = mo.ui.number(0, 20000, value=1000, step=250, label="Lambda")
    adv_var_spike = mo.ui.number(
        0,
        10,
        value=0.1,
        step=0.005,
        label="Spike distribution variance",
    )
    adv_var_slab = mo.ui.number(
        0,
        1000,
        value=100,
        step=10,
        label="Slab distribution variance",
    )

    # Parameter help text
    parameter_help = mo.accordion(
        {
            "📖 Parameter guide": mo.md(
                """
                ### Basic parameters

                - **Number of components:** How many distinct feature sets to discover. It is recommended to overshoot this number (2-3x), especially in adverse conditions.
                - **Estimated sparsity:** Expected number of features per solution. This guides the algorithm's search.

                ### Advanced optimization settings

                - **Iterations:** More iterations improve convergence but take longer (typical: 3000-5000). Increase if ELBO hasn't converged or features' mu values are still changing.
                - **Learning rate:** Controls [SGD optimization](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) step size (typical: 0.001-0.003). Decrease if training is unstable, increase if progress is too slow.
                - **Batch size:** Number of samples used in one SGD optimization step (typical: 16-64). Increase for datasets with missing data, noise, or class imbalance. Increasing batch size proportionally increases run time. Recommendation: have at least 4 samples of the minority class in one batch.
                - **Enforce diversity:** Penalizes average similarity ([Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)) of solutions to promote diverse feature combinations. Enable when you want to push more towards distinct explanatory mechanisms.
                - **Lambda:** Strength of diversity penalty (typical: 0-2000). Higher values → more different solutions. Increase if solutions overlap too much.

                <i>**Example batch size setup.**
                Let your dataset contain 200 samples: 160x class A, 40x class B (i.e. the minority class makes up 20% samples). There is low noise and no missing data.
                It is desirable to have *at least* 4 samples from each class in a batch (empirical observation) => batch size = 4 * (1/0.20) = 20 samples.
                </i>

                ### Advanced prior settings

                These control how the algorithm balances sparsity (few features) vs. explanatory power (including relevant features).

                This algorithm uses the [Structured Spike-and-Slab](https://en.wikipedia.org/wiki/Spike-and-slab_regression) [prior distribution](https://en.wikipedia.org/wiki/Prior_probability),
                that is a mixture of two Gaussian distributions.
                Each feature is assigned to either the wide distribution (Slab) or the steep distribution (Spike).

                - **Spike variance:**
                    - Is the most important parameter for controlling convergence.
                    - Controls sparsity strength (typical: 0.05-0.5).
                    - Increase if all features converge to 0 (over-regularization).
                    - Decrease carefully if too many features are selected (under-regularization).

                - **Slab variance:**
                    - Scale for non-zero features (typical: 50-200).
                    - Adjust together with spike variance to improve feature discrimination.
                """
            )
        }
    )

    settings_ui = mo.vstack(
        [
            parameter_help,
            mo.md("<br>"),
            mo.vstack([n_candidates, sparsity_est]),
            mo.accordion(
                {
                    "Advanced optimization settings": mo.vstack(
                        [adv_iter, adv_lr, adv_batch, adv_jaccard, adv_lambda]
                    )
                }
            ),
            mo.accordion(
                {
                    "Advanced prior settings (Structured Spike-and-Slab)": mo.vstack(
                        [adv_var_spike, adv_var_slab]
                    )
                }
            ),
        ]
    )
    settings_ui
    return (
        adv_batch,
        adv_iter,
        adv_jaccard,
        adv_lambda,
        adv_lr,
        adv_var_slab,
        adv_var_spike,
        n_candidates,
        sparsity_est,
    )


@app.cell
def _(df_processed, mo):
    mo.stop(df_processed is None, "")

    # The big RUN FEATURE SELECTION button
    run_btn = mo.ui.run_button(label="Run feature selection", kind="success")

    mo.vstack(
        [
            run_btn,
            mo.md("<br>"),
            mo.md("---"),
        ]
    )
    return (run_btn,)


@app.cell
def _(
    BayesianFeatureSelector,
    X,
    adv_batch,
    adv_iter,
    adv_jaccard,
    adv_lambda,
    adv_lr,
    adv_var_slab,
    adv_var_spike,
    df_raw,
    mo,
    n_candidates,
    run_btn,
    sparsity_est,
    y,
):
    # Main execution logic

    # 1. Stop if data not loaded
    mo.stop(df_raw is None, mo.md("*Please upload data first.*"))

    # 2. Stop if button not pressed
    mo.stop(not run_btn.value, mo.md("*Ready to run. Click start above.*"))

    # Optimization: class setup
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=n_candidates.value,
        X=X,
        y=y,
        prior="sss",
        sss_sparsity=sparsity_est.value,
        var_slab=adv_var_slab.value,
        var_spike=adv_var_spike.value,
        lr=adv_lr.value,
        batch_size=adv_batch.value,
        n_iter=adv_iter.value,
    )

    # Run optimizer
    history = selector.optimize(
        regularize=adv_jaccard.value,
        lambda_jaccard=adv_lambda.value,
        verbose=True,
    )
    return (history,)


@app.cell
def _(
    X,
    adv_batch,
    adv_iter,
    adv_jaccard,
    adv_lambda,
    adv_lr,
    adv_var_slab,
    adv_var_spike,
    n_candidates,
    os,
    save_dir_input,
    save_experiment_id,
    save_features_name,
    save_history_name,
    save_results,
    save_setup_name,
    sparsity_est,
):
    # Saving setup
    if save_results:
        # Configure saving options, if saving is enabled
        # Prepare directory
        experiment_dir = f"{save_dir_input.value}/experiment_{save_experiment_id.value}"
        os.makedirs(experiment_dir, exist_ok=True)

        # Prepare save paths
        history_path = f"{experiment_dir}/{save_history_name.value}.json"
        setup_path = f"{experiment_dir}/{save_setup_name.value}.json"
        features_path_json = f"{experiment_dir}/{save_features_name.value}.json"
        features_path_txt = f"{experiment_dir}/{save_features_name.value}.txt"

        # Define constants that are to be saved
        constants = {
            "N_SAMPLES": X.shape[0],
            "N_FEATURES": X.shape[1],
            "N_CANDIDATE_SOLUTIONS": n_candidates.value,
            "SPARSITY": sparsity_est.value,
            "PRIOR_SPARSITY": sparsity_est.value,
            "PRIOR_TYPE": "sss",
            "VAR_SPIKE": adv_var_spike.value,
            "VAR_SLAB": adv_var_slab.value,
            "N_ITER": adv_iter.value,
            "LEARNING_RATE": adv_lr.value,
            "BATCH_SIZE": adv_batch.value,
            "IS_REGULARIZED": adv_jaccard.value,
            "LAMBDA_JACCARD": adv_lambda.value,
        }
    else:
        experiment_dir = None
        history_path = None
        setup_path = None
        features_path_json = None
        features_path_txt = None
    return (
        constants,
        experiment_dir,
        features_path_json,
        features_path_txt,
        history_path,
        setup_path,
    )


@app.cell
def _(
    constants,
    experiment_dir,
    history,
    history_path,
    mo,
    save_constants_json,
    save_results,
    save_selector_history_json,
    setup_path,
):
    # Save history and setup immediately after optimization
    if save_results:
        # Save history and constants
        msg_history = save_selector_history_json(history, history_path)
        msg_constants = save_constants_json(constants, setup_path)

        # Stack output messages
        _display = mo.vstack(
            [
                mo.md("✅ **Optimization Complete!**"),
                mo.md(f"📁 Optimization history saved to: `{experiment_dir}`"),
                mo.md(f"- {msg_history}"),
                mo.md(f"- {msg_constants}"),
                mo.md("<br>"),
                mo.md("---"),
                mo.md("<br>"),
            ]
        )
    else:
        _display = mo.vstack(
            [
                mo.md("✅ **Optimization Complete!**"),
                mo.md("<br>"),
                mo.md("---"),
                mo.md("<br>"),
            ]
        )

    _display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **3. Algorithm progress history**

    Assess convergence and features in the components. If needed, adjust the algorithm's parameters and rerun.
    """
    )
    return


@app.cell
def _(
    adv_iter,
    feature_map,
    get_algorithm_progress_plots,
    get_final_alphas_plot,
    history,
    mo,
    n_features,
):
    mo.stop(history is None, "")  # Show only after feature selector is run

    # Visualization of algorithm trajectories

    # Get progress plots
    progress_plots_dict = get_algorithm_progress_plots(
        history,
        elbo=True,
        mu=True,
        alpha=False,
        original_feature_names_mapping=feature_map,
        subsample_history_for_plotting=(
            True if ((adv_iter.value > 4000) or (n_features > 80)) else False
        ),
    )

    alphas_plots = get_final_alphas_plot(
        history,
        show_bar_plot=False,
        show_pie_chart=True,
    )
    alpha_piechart = alphas_plots[0]

    # Elbo convergence help text
    elbo_help = mo.accordion(
        {
            " 📖 Guide": mo.md(
                r"""
                The [ELBO (Evidence Lower BOund)](https://en.wikipedia.org/wiki/Evidence_lower_bound) is the objective function that the algorithm maximizes, possibly combined with penalization: $ELBO - \lambda * penalty$

                The objective function's value should steadily increase and eventually plateau just below zero.
                Note that oscillations occur naturally due to inherent stochasticity.

                **What to look for:**
                - **Good convergence:** Steady upward trend that flattens into a plateau. The curve should stabilize before the end of iterations. The final 10-20% of iterations should show minimal change.
                - **Not converged:** Still increasing steeply at the end → increase number of iterations.
                - **Unstable:** Erratic oscillations are not dampened over time → decrease learning rate or adjust batch size.
                - **Too high absolute value:** Values in millions or more signify a major problem with the algorithm setup. Assess the features' trajectories below. (Values in low thousands, depending on lambda, are generally favorable.)
                """
            )
        }
    )

    # Feature convergence help text
    mu_help = mo.accordion(
        {
            " 📖 Guide": mo.md(
                """
                **These plots are crucial for assessment of whether the feature selector produced sensible results.**

                Each plot shows how feature importance values (mu) evolve during optimization for one component.
                The algorithm assigns each feature to either the Spike (near zero) or Slab (non-zero) distribution.

                **What to look for:**
                - **Good separation:** Clear gap between features converging to ~0 and features with significant non-zero values.
                - **Optimization process:** 

                **Problem indicators:**
                - **All features → 0:** Over-regularization. Increase spike variance.
                - **Features' ordering does not change over time:** May indicate under-regularization.
                - **Too many non-zero features:** May indicate under-regularization. Carefully decrease spike variance and/or enforce stronger sparsity.
                - **Oscillating values:** Learning rate too high or batch size too small.
                - **Set of important features changes multiple times over time** (situation: while most features converge to 0, a few become nonzero. Then the nonzero features become 0 and others emerge. The set of important features significantly changes multiple times.): The algorithm cannot find any significant signal among noise. May indicate fundamental problem with the dataset.
                """
            )
        }
    )

    # Alpha help text
    alpha_help = mo.accordion(
        {
            " 📖 Guide": mo.md(
                """
                The pie chart shows how the algorithm distributes probability mass across the components.
                (Alphas represent the mixing weights in the [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) that approximate the [posterior distribution](https://en.wikipedia.org/wiki/Posterior_probability).)

                The alphas should correspond to the predictive potential of the components.

                **What to look for:**
                - **(Un)balanced distribution:** If you expect multiple solutions of comparable significance, alphas should be relatively balanced across components. If some components dominate, too many components may have been requested (i.e. the dataset supports fewer distinct solutions).
                """
            )
        }
    )

    mo.vstack(
        [
            mo.md("<br>"),
            mo.md("### 3.1 Objective function convergence"),
            elbo_help,
            progress_plots_dict["elbo"].update_layout(
                height=400, width=200 + adv_iter.value / 8
            ),
            mo.md("<br>"),
            mo.md("### 3.2 Feature convergence in components"),
            mu_help,
            # Unpack all mu plots
            *[
                progress_plots_dict[_plot].update_layout(
                    height=400, width=400 + adv_iter.value / 8
                )
                for _plot in progress_plots_dict.keys()
                if "mu_" in _plot
            ],
            mo.md("<br>"),
            mo.md("### 3.3 Relative importances of components"),
            alpha_help,
            alpha_piechart.update_layout(height=450, width=450),
            mo.md("<br>"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **4. Recover solutions from components**

    Each component can be handled in multiple ways to yield feature sets = candidate solutions. Select your strategy.
    """
    )
    return


@app.cell
def _(df_processed, history, mo, sparsity_est):
    mo.stop(history is None, "")  # Show only after feature selector is run

    # checkboxes to pick solution types
    checkbox_out20_sol = mo.ui.checkbox(label="Outliers with STD > 2.0", value=True)
    checkbox_out25_sol = mo.ui.checkbox(label="Outliers with STD > 2.5", value=True)
    checkbox_out30_sol = mo.ui.checkbox(label="Outliers with STD > 3.0", value=True)
    checkbox_out35_sol = mo.ui.checkbox(label="Outliers with STD > 3.5", value=False)
    checkbox_top_sol = mo.ui.checkbox(label="Top few features", value=False)
    checkbox_full_sol = mo.ui.checkbox(
        label="All features with mu > threshold", value=False
    )

    # advanced settings
    top_n_features_selector = mo.ui.number(
        1,
        df_processed.shape[1],
        value=sparsity_est.value,
        label="Hard number of top features to be selected ('top' solution type).",
    )
    min_mu_selector = mo.ui.number(
        0,
        5,
        value=0.25,
        label="'Zero' threshold for mu values",
    )

    # display options
    checkbox_summary = mo.ui.checkbox(label="Summary", value=True, disabled=True)
    checkbox_matrix = mo.ui.checkbox(
        label="Feature distribution across components", value=True
    )
    checkbox_regression_l2 = mo.ui.checkbox(
        label="Regression with l2 regularization", value=True
    )
    checkbox_regression_l1 = mo.ui.checkbox(
        label="Regression with l1 regularization", value=False
    )

    # description of solution types
    solution_recovery_help = mo.accordion(
        {
            " 📖 Guide": mo.md(
                """
                ### Solution recovery strategies

                Each component from the feature selector contains information about feature importance (mu values). These can be converted into feature sets (candidate solutions) using different strategies:

                **Outlier-based solutions** (recommended)

                Identifies features with unusually high |mu| values using [statistical outlier detection](https://en.wikipedia.org/wiki/Outlier). Features are selected if their |mu| exceeds:
                - **STD > 2.0**: Most inclusive (2 standard deviations from mean)
                - **STD > 2.5**: Moderately selective
                - **STD > 3.0**: More selective (common choice)
                - **STD > 3.5**: Most restrictive

                **Top-few solutions**

                Selects a fixed number of top features with highest |mu| values.
                The number is specified in the advanced settings below.

                **Thresholded solutions**

                Selects all features with |mu| values above an arbitrary threshold (defined in the advanced settings below).
                This approach is useful for further assessment of importance of the features selected in other strategies 
                because it can order all features in a component based on their |mu| values.
                For example, if a feature has a significant |mu| value in only one component with generously set threshold, it is likely noise.
                Important features usually have significant |mu| values in multiple components.
                """
            )
        }
    )

    # description of solution display options
    solution_display_help = mo.accordion(
        {
            " 📖 Guide": mo.md(
                """
                For each solution type, you can display:

                **Summary** (always shown): Table listing all features in each component with their mu values, sorted by importance. Helps you understand which features the algorithm selected and their relative importance in the model.

                **Feature distribution across components**: Heatmap/matrix showing which features appear in which components. Useful for:
                - Identifying features that appear across multiple components (potentially robust predictors)
                - Checking solution diversity (do different components select different features?)
                - Understanding feature overlap patterns

                **Regression validation**: Quick assessment of each solution's predictive performance using simple linear/logistic regression:
                - **L2 regularization** (Ridge): Handles correlated features well, generally more stable
                - **L1 regularization** (Lasso): Performs additional feature selection, may be more interpretable

                ⚠️ **Important:** These regression metrics use the *same data for training and testing* (no cross-validation), so they provide only a preliminary quality check. For rigorous evaluation, create full models.

                The regression results help you quickly identify which solutions show promise before investing time in full model evaluation.
                """
            )
        }
    )

    mo.vstack(
        [
            mo.md("### Pick solution types to be recovered from components:"),
            solution_recovery_help,
            mo.vstack(
                [
                    checkbox_out20_sol,
                    checkbox_out25_sol,
                    checkbox_out30_sol,
                    checkbox_out35_sol,
                    checkbox_top_sol,
                    checkbox_full_sol,
                ]
            ),
            mo.accordion(
                {
                    "Advanced setting": mo.vstack(
                        [top_n_features_selector, min_mu_selector]
                    )
                }
            ),
            mo.md("<br>"),
            mo.md("### Pick what is to be shown:"),
            solution_display_help,
            mo.vstack(
                [
                    checkbox_summary,
                    checkbox_matrix,
                    checkbox_regression_l2,
                    checkbox_regression_l1,
                ]
            ),
        ]
    )
    return (
        checkbox_full_sol,
        checkbox_matrix,
        checkbox_out20_sol,
        checkbox_out25_sol,
        checkbox_out30_sol,
        checkbox_out35_sol,
        checkbox_regression_l1,
        checkbox_regression_l2,
        checkbox_top_sol,
        min_mu_selector,
        top_n_features_selector,
    )


@app.cell
def _(history, mo):
    mo.stop(history is None, "")  # Show only after feature selector is run

    # The big RUN button
    recover_btn = mo.ui.run_button(label="Recover solutions", kind="success")

    mo.vstack(
        [
            mo.md("<br>"),
            recover_btn,
            mo.md("<br>"),
            mo.md("---"),
        ]
    )
    return (recover_btn,)


@app.cell
def _(
    checkbox_full_sol,
    checkbox_matrix,
    checkbox_out20_sol,
    checkbox_out25_sol,
    checkbox_out30_sol,
    checkbox_out35_sol,
    checkbox_regression_l1,
    checkbox_regression_l2,
    checkbox_top_sol,
    detect_task,
    df_raw,
    experiment_dir,
    feature_map,
    features_path_json,
    features_path_txt,
    get_features_from_solutions,
    get_features_in_components_plot,
    get_solution_summary_df,
    get_unique_features,
    history,
    min_mu_selector,
    mo,
    recover_btn,
    recover_solutions,
    save_feature_lists_json,
    save_feature_lists_txt,
    save_results,
    scaling_selector,
    solve_any_regression,
    top_n_features_selector,
    y,
):
    mo.stop(history is None, "")  # Show only after feature selector is run

    mo.stop(
        not recover_btn.value,
        mo.md("*Ready to recover solutions from components. Click button above.*"),
    )
    mo.stop(
        (top_n_features_selector.value is None) or (top_n_features_selector.value < 1),
        mo.md("Please set 'top few features' to value 1 or more."),
    )

    # Define which outliers are to be recovered
    outlier_deviation_thresholds = []
    if checkbox_out20_sol.value:
        outlier_deviation_thresholds.append(2.0)
    if checkbox_out25_sol.value:
        outlier_deviation_thresholds.append(2.5)
    if checkbox_out30_sol.value:
        outlier_deviation_thresholds.append(3.0)
    if checkbox_out35_sol.value:
        outlier_deviation_thresholds.append(3.5)

    # Recover solutions
    sol_full, sol_top, sol_outliers, _ = recover_solutions(
        search_history=history,
        desired_sparsity=top_n_features_selector.value,
        min_mu_threshold=min_mu_selector.value,
        original_feature_names_mapping=feature_map,
        use_median_for_outlier_detection=False,
        outlier_deviation_thresholds=outlier_deviation_thresholds,
    )

    # Put all the requested solution types into a single dictionary
    all_solutions = {}
    for _key, _outlier in sol_outliers.items():
        all_solutions[f"Outlier features ({' = '.join(_key.split(sep='_'))})"] = (
            _outlier
        )
    if checkbox_top_sol.value:
        all_solutions["Top features"] = sol_top
    if checkbox_full_sol.value:
        all_solutions["Thresholded features"] = sol_full

    mo.stop(
        all_solutions == {},
        mo.md("Cannot proceed. Please pick a solution type to recover."),
    )

    # Extract which features are contained in which solution type
    # Get overviews and simple performance metrics
    solution_summary = (
        {}
    )  # one dateframe per solution type: feature names with mu values
    all_feature_sets = {}  # features per component, for each solution type
    unique_features_found = (
        {}
    )  # all unique features across all components of a solution type
    regression_metrics_l1 = {}
    regression_metrics_l2 = {}

    for _type, _solution in all_solutions.items():
        solution_summary[_type] = get_solution_summary_df(_solution)
        all_feature_sets[_type] = get_features_from_solutions(_solution)
        unique_features_found[_type] = get_unique_features(all_feature_sets[_type])

        # Quick validation with simple linear/logistic regression
        # l2-regularized
        if checkbox_regression_l2.value and (df_raw is not None):
            regression_metrics_l2[_type] = solve_any_regression(
                solutions=all_feature_sets[_type],
                df=df_raw,  # df_processed,
                response=y,
                apply_scaling=scaling_selector.value,
                penalty="l2",
                verbose=False,
            )
        # l1-regularized
        if checkbox_regression_l1.value and (df_raw is not None):
            regression_metrics_l1[_type] = solve_any_regression(
                solutions=all_feature_sets[_type],
                df=df_raw,  # df_processed,
                response=y,
                apply_scaling=scaling_selector.value,
                penalty="l1",
                verbose=False,
            )

    if save_results:
        # Save candidate solutions
        msg_features_json = save_feature_lists_json(
            all_feature_sets, features_path_json
        )
        msg_features_txt = save_feature_lists_txt(all_feature_sets, features_path_txt)

        # Stack all the outputs in the correct order
        _displays = [
            mo.md(f"📁 **All recovered solutions saved to:** `{experiment_dir}`"),
            mo.md(f"- {msg_features_txt.split('Candidate solutions saved to ')[1]}"),
            mo.md(f"- {msg_features_json.split('Candidate solutions saved to ')[1]}"),
            mo.md("---"),
            mo.md("<br><br>"),
        ]
    else:
        _displays = []

    for _type, _solution in all_solutions.items():
        # Get summary of a solution type
        _displays.append(mo.md(f"### Solution type: **{_type}**"))
        _displays.append(mo.ui.table(solution_summary[_type]))

        # Get a matrix of features vs. components
        if checkbox_matrix.value:
            _displays.append(
                get_features_in_components_plot(
                    solutions=all_feature_sets[_type],
                    features_to_show=unique_features_found[_type],
                ).update_layout(showlegend=False),
            )

        # Regression or classification?
        task_type = detect_task(y)

        # Get quick validation with a simple regression
        if checkbox_regression_l2.value or checkbox_regression_l1.value:
            regression_type = "logistic" if task_type == "classification" else "linear"

        # l2-regularized
        if checkbox_regression_l2.value:
            _displays.append(
                mo.md(
                    f"#### **Quick l2-regularized {regression_type} regression validation** for {_type} (testing = training data):"
                )
            )
            _displays.append(mo.ui.table(regression_metrics_l2[_type]))

        # l1-regularized
        if checkbox_regression_l1.value:
            _displays.append(
                mo.md(
                    f"#### **Quick l1-regularized {regression_type} regression validation** for {_type} (testing = training data):"
                )
            )
            _displays.append(mo.ui.table(regression_metrics_l1[_type]))

        _displays.append(mo.md("<br><br>"))
        _displays.append(mo.md("---"))
        _displays.append(mo.md("<br><br>"))

    # Return all displays stacked vertically
    mo.vstack(_displays)
    return all_feature_sets, all_solutions, task_type, unique_features_found


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **5. Modeling with candidate solutions** [non-commercial use only]

    Using an advanced algorithm to create and evaluate models for each feature set of the chosen solution type. Proper train-test cross-validation is run.

    **WARNING:** For downstream modeling, we use TabPFN, whose free licence can be used only for research purposes. [(Read more.)](https://huggingface.co/Prior-Labs/tabpfn_2_5)
    """
    )
    return


@app.cell
def _(all_solutions, mo):
    # pick one solution type for further evaluation
    mo.stop(
        all_solutions is None,
        mo.md("*Must recover solutions from components first. Click button above.*"),
    )

    radio_solutions = mo.ui.radio(
        options=all_solutions.keys(),
        label="Choose one solution type:",
    )

    mo.vstack(
        [
            radio_solutions,
        ]
    )
    return (radio_solutions,)


@app.cell
def _(mo, radio_solutions):
    mo.stop(
        radio_solutions.value is None,
        output=mo.md("*Cannot proceed to modeling. Pick a solution type.*"),
    )

    # The big RUN button
    model_btn = mo.ui.run_button(
        label="Model with TabPFN (non-commercial use only)", kind="success"
    )
    checkbox_shap = mo.ui.checkbox(
        value=False,
        label="Compute feature importances in the model (Shapley values).",
    )

    mo.vstack(
        [
            mo.md("<br>"),
            checkbox_shap,
            model_btn,
            mo.md("<br>"),
            mo.md("---"),
        ]
    )
    return checkbox_shap, model_btn


@app.cell
def _(
    all_feature_sets,
    all_solutions,
    checkbox_shap,
    df_raw,
    experiment_dir,
    mo,
    model_btn,
    pd,
    radio_solutions,
    save_results,
    tabpfn_evaluate,
    task_type,
    unique_features_found,
    y,
):
    mo.stop(
        not model_btn.value,
        output=mo.md("*Solutions prepared. Press button to start modeling.*"),
    )

    # Get the selected solution type
    selected_solution_type = radio_solutions.value
    selected_solution = all_solutions[selected_solution_type]

    _eval_displays = []
    _eval_displays.append(
        mo.md(
            f"""
            ### Evaluating: **{selected_solution_type}**
            - Task type: **{task_type}**
            - Number of components: **{len(selected_solution)}**
            - Computing SHAP explanations: **{"Yes" if checkbox_shap.value else "No"}**
            """
        )
    )

    # Get features for each component
    component_features = all_feature_sets[selected_solution_type]
    component_features["all_selected_features"] = unique_features_found[
        selected_solution_type
    ]

    # Evaluate each component with TabPFN
    tabpfn_results = {}
    for component_name, feature_list in component_features.items():
        # Get the feature subset
        X_component = df_raw[feature_list]  # df_processed[feature_list]

        # Run TabPFN evaluation
        tabpfn_results[component_name] = tabpfn_evaluate(
            X_component,
            y,
            apply_scaling=None,  # Already scaled during preprocessing
            outer_cv_folds=2,
            tabpfn_kwargs=None,
            random_state=42,
            verbose=False,
            explain=(
                checkbox_shap.value
                if component_name != "all_selected_features"
                else False
            ),
            shap_sample_size=50,
        )

    # Show average scores (exclude non-scalar values)
    all_scores = pd.DataFrame(
        {
            _comp: {
                k: v
                for k, v in _result["average_scores"].items()
                if k not in ["confusion_matrix_sum", "class_distribution"]
            }
            for _comp, _result in tabpfn_results.items()
        }
    )
    _eval_displays.append(mo.ui.table(all_scores))

    # Show Shapley values if computed
    for _comp, _result in tabpfn_results.items():
        if "shap_explanations" in _result:
            _eval_displays.append(
                mo.md(
                    f"**Shapley values:** feature importances in the model from {_comp}"
                )
            )
            _eval_displays.append(
                mo.ui.table(pd.DataFrame(_result["shap_explanations"]))
            )
            _eval_displays.append(mo.md("<br>"))

    # Save results if enabled
    if save_results:
        scores_path = f"{experiment_dir}/tabpfn_scores_{selected_solution_type.replace(' ', '_')}.csv"
        all_scores.to_csv(scores_path)
        _eval_displays.append(
            mo.md(f"📊 **Results saved to:** `{scores_path.split('/')[-1]}`<br><br>"),
        )

    mo.vstack(_eval_displays)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
