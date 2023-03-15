import streamlit as st
from streamlit_ace import st_ace

from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType

from callbacks import analyze_pipeline, get_report
from constants import PIPELINE_CONFIG


ANALYSIS_RESULT = None


st.set_page_config(page_title="mlwhatif", page_icon="🧐", layout="wide")
st.title("`mlwhatif` demo")
# with st.echo():
#     st.__version__


### === SIDEBAR / CONFIGURATION ===
st.sidebar.title("Menu")

# Pipeline
pipeline = st.sidebar.selectbox("Choose a pipeline", list(PIPELINE_CONFIG.keys()))
pipeline_filename = PIPELINE_CONFIG[pipeline]["filename"]
pipeline_columns = PIPELINE_CONFIG[pipeline]["columns"]

# What-if Analyses
analyses = {}
if st.sidebar.checkbox("Data Corruption"):  # a.k.a. robustness
    # column_to_corruption: List[Tuple[str, Union[FunctionType, CorruptionType]]],
    column_to_corruption = {}
    selected_columns = st.sidebar.multiselect(
        "Columns to corrupt", pipeline_columns)
    for column in selected_columns:
        column_to_corruption[column] = st.sidebar.selectbox(
            column, CorruptionType.__members__.values(), format_func=lambda m: m.value)

    # corruption_percentages: Iterable[Union[float, Callable]] or None = None,
    corruption_percentages = []
    num = st.sidebar.number_input(
        "Corruption percentages", min_value=0.0, max_value=1.0, step=0.01, key=0)
    while num and num > 0:
        corruption_percentages.append(num)
        num = st.sidebar.number_input("Corruption percentages", min_value=0.0,
                                      max_value=1.0, step=0.01, key=len(corruption_percentages))

    # also_corrupt_train: bool = False):
    also_corrupt_train = st.sidebar.checkbox("Also corrupt train")

    # __init__
    robustness = DataCorruption(column_to_corruption=list(column_to_corruption.items()),
                                corruption_percentages=corruption_percentages,
                                also_corrupt_train=also_corrupt_train)
    analyses["robustness"] = robustness

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
    if pipeline_filename:
        with open(pipeline_filename) as f:
            pipeline_code = f.read()
        with pipeline_code_container:
            # st.code(pipeline_code)
            # Check out more themes: https://github.com/okld/streamlit-ace/blob/main/streamlit_ace/__init__.py#L36-L43
            final_pipeline_code = st_ace(value=pipeline_code, language="python", theme="katzenmilch")

with right:
    if run_button:
        with analysis_results_container:
            with st.spinner():
                ANALYSIS_RESULT = analyze_pipeline(pipeline_filename, *analyses.values())
            st.balloons()

            for analysis in analyses.values():
                report = get_report(ANALYSIS_RESULT, analysis)
                st.subheader(analysis.__class__.__name__)
                st.table(report)

    if ANALYSIS_RESULT:
        with optimized_dag_container:
            st.write("optimized DAG placeholder")
            ANALYSIS_RESULT.combined_optimized_dag

with left:
    if ANALYSIS_RESULT:
        with original_dag_container:
            st.write("original DAG placeholder")
            ANALYSIS_RESULT.original_dag
