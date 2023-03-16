import streamlit as st
from fairlearn.metrics import MetricFrame
from streamlit_ace import st_ace

from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.analysis._permutation_feature_importance import PermutationFeatureImportance
from mlwhatif.analysis._operator_impact import OperatorImpact
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType

from callbacks import analyze_pipeline, get_report, render_graph1, render_graph2
from constants import PIPELINE_CONFIG


ANALYSIS_RESULT = None


st.set_page_config(page_title="mlwhatif", page_icon="🧐", layout="wide")
st.title("`mlwhatif` demo")
# with st.echo():
#     st.__version__


### === SIDEBAR / CONFIGURATION ===
st.sidebar.title("Configuration")

# Pipeline
pipeline = st.sidebar.selectbox("Choose a pipeline", list(PIPELINE_CONFIG.keys()))
pipeline_filename = PIPELINE_CONFIG[pipeline]["filename"]
pipeline_columns = PIPELINE_CONFIG[pipeline]["columns"]
if pipeline_filename:
    with open(pipeline_filename) as f:
        pipeline_code = f.read()
    pipeline_num_lines = len(pipeline_code.splitlines())

# What-if Analyses
analyses = {}

if st.sidebar.checkbox("Data Corruption"):  # a.k.a. robustness
    # column_to_corruption: List[Tuple[str, Union[FunctionType, CorruptionType]]],
    column_to_corruption = {}
    selected_columns = st.sidebar.multiselect("Columns to corrupt", pipeline_columns)
    for column in selected_columns:
        column_to_corruption[column] = st.sidebar.selectbox(
            column, CorruptionType.__members__.values(), format_func=lambda m: m.value)

    # corruption_percentages: Iterable[Union[float, Callable]] or None = None,
    corruption_percentages = []
    num = st.sidebar.number_input(
        "Corruption percentage", min_value=0.0, max_value=1.0, step=0.01, key=0)
    while num and num > 0:
        corruption_percentages.append(num)
        num = st.sidebar.number_input("Corruption percentage", min_value=0.0,
                                      max_value=1.0, step=0.01, key=len(corruption_percentages))

    # also_corrupt_train: bool = False):
    also_corrupt_train = st.sidebar.checkbox("Also corrupt train")

    # __init__
    robustness = DataCorruption(column_to_corruption=list(column_to_corruption.items()),
                                corruption_percentages=corruption_percentages,
                                also_corrupt_train=also_corrupt_train)
    analyses["robustness"] = robustness

# if st.sidebar.checkbox("Permutation Importance"):
#     # restrict_to_columns: Iterable[str] or None = None
#     restrict_to_columns = st.sidebar.multiselect("Restrict to columns", pipeline_columns)
#
#     # __init__
#     importance = PermutationFeatureImportance(restrict_to_columns=restrict_to_columns)
#     analyses["importance"] = importance

if st.sidebar.checkbox("Operator Impact"):
    # test_transformers=True
    test_transformers = st.sidebar.checkbox("Test selections", value=True)

    # test_selections=False
    test_selections = st.sidebar.checkbox("Test selections")

    # restrict_to_linenos: List[int] or None = None
    # line_numbers = []
    # num = st.sidebar.number_input(
    #     "Line number", min_value=1, max_value=pipeline_num_lines, step=1, key=0)
    # while num and num > 0:
    #     line_numbers.append(num)
    #     num = st.sidebar.number_input("Line number", min_value=1, max_value=pipeline_num_lines,
    #                                   step=1, key=len(line_numbers))

    # __init__
    preproc = OperatorImpact(test_transformers=test_transformers, test_selections=test_selections)
    analyses["preproc"] = preproc

if st.sidebar.checkbox("Data Cleaning"):
    # columns_with_error: dict[str or None, ErrorType] or List[Tuple[str, ErrorType]]
    columns_with_error = {}
    selected_columns = st.sidebar.multiselect("Columns with errors", pipeline_columns + ["_TARGET_"])
    for column in selected_columns:
        columns_with_error[column] = st.sidebar.selectbox(
            column, ErrorType.__members__.values(), format_func=lambda m: m.value)

    # __init__
    cleanlearn = DataCleaning(columns_with_error=columns_with_error)
    analyses["cleanlearn"] = cleanlearn

# Actions
scan_button = st.sidebar.button("Scan Pipeline")
estimate_button = st.sidebar.button(
    "Estimate Execution Time", disabled=not scan_button)  # estimate execution time
run_button = st.sidebar.button("Run Analyses")


### === LAYOUT ===
left, right = st.columns(2)

with left:
    pipeline_code_container = st.expander("Pipeline Code", expanded=True)
    original_dag_container = st.expander("Original DAG")

with right:
    analysis_results_container = st.expander("Analysis Results")
    optimized_dag_container = st.expander("Optimized DAG")

### === MAIN CONTENT ===
with left:
    with pipeline_code_container:
        # st.code(pipeline_code)
        # Check out more themes: https://github.com/okld/streamlit-ace/blob/main/streamlit_ace/__init__.py#L36-L43
        final_pipeline_code = st_ace(value=pipeline_code, language="python", theme="katzenmilch")

with right:
    if run_button:
        with analysis_results_container:
            with st.spinner("Analyzing pipeline..."):
                ANALYSIS_RESULT = analyze_pipeline(pipeline_filename, *analyses.values())
            st.balloons()

            for analysis in analyses.values():
                report = get_report(ANALYSIS_RESULT, analysis)

                metrics_frame_columns = report.select_dtypes('object')
                for column in metrics_frame_columns:
                    if len(report) != 0 and isinstance(report[column].iloc[0], MetricFrame):
                        # TODO: Better visualisation or remove MetricFrame from healthcare pipeline
                        report[column] = report.apply(lambda row: str(row[column].by_group), axis=1)

                st.subheader(analysis.__class__.__name__)
                st.table(report)

    if ANALYSIS_RESULT:
        with optimized_dag_container:
            render_graph2(ANALYSIS_RESULT.combined_optimized_dag)

with left:
    if ANALYSIS_RESULT:
        with original_dag_container:
            render_graph2(ANALYSIS_RESULT.original_dag)
