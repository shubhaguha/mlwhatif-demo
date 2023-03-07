from functools import partial
import os

import streamlit as st
from sklearn.linear_model import LogisticRegression

from demo.feature_overview.clean_learn import Clean, ErrorType, CleanLearn
from demo.feature_overview.operator_impact import OperatorImpact
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.utils import get_project_root


st.set_page_config(page_title="mlwhatif", page_icon="🧐")

### --- Sidebar ---
st.sidebar.title("Menu")

# Pipeline
pipeline = st.sidebar.selectbox("Choose a pipeline", ("healthcare", "reviews", "pipeline3"))

if pipeline == "healthcare":
    pipeline_filename = os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.py")
elif pipeline == "reviews":
    pipeline_filename = os.path.join(str(get_project_root()), "demo", "advanced_features", "reviews.py")

# What-if Analyses
selected_analyses = []

if st.sidebar.checkbox("CleanLearn"):
    cleanlearn = CleanLearn(column="weight",
                            error=ErrorType.OUTLIER,
                            cleanings=[Clean.FILTER, Clean.IMPUTE],
                            outlier_func=lambda y: (y > 120) | (y < 30),
                            impute_constant=70)
    selected_analyses.append(cleanlearn)

if st.sidebar.checkbox("OperatorImpact"):
    op_impact = OperatorImpact(robust_scaling=True,
                               named_model_variants=[('logistic_regression', partial(LogisticRegression))])
    selected_analyses.append(op_impact)

# Buttons
scan_button = st.sidebar.button("Scan Pipeline")
estimate_button = st.sidebar.button("Estimate Execution Time", disabled=not scan_button)  # estimate execution time
run_button = st.sidebar.button("Run Analyses")

### --- Main content ---
st.title("`mlwhatif` demo")

if pipeline_filename:
    st.header("Pipeline Code")
    with open(pipeline_filename) as f:
        st.code(f.read())
        # TODO: Add line numbers

def analyze_pipeline(pipeline_filename, *_what_if_analyses):
    builder = PipelineAnalyzer.on_pipeline_from_py_file(pipeline_filename)

    for analysis in _what_if_analyses:
        builder = builder.add_what_if_analysis(analysis)

    builder = builder.add_custom_monkey_patching_module(custom_monkeypatching)
    analysis_result = builder.execute()

    return analysis_result


def get_report(result, what_if_analysis):
    report = result.analysis_to_result_reports[what_if_analysis]
    return report


if run_button:
    st.header("Analysis Results")
    st.spinner()
    result = analyze_pipeline(pipeline_filename, *selected_analyses)

    for analysis in selected_analyses:
        report = get_report(result, analysis)
        st.subheader(analysis.__class__.__name__)
        st.table(report)
