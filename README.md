`mlwhatif` demo
===

This web app was created to demonstrate the functionality of `mlwhatif`: <https://github.com/stefan-grafberger/mlwhatif>

`mlwhatif` is a tool for Data-Centric What-If Analysis for Native Machine Learning Pipelines. It uses the [`mlinspect`](https://github.com/stefan-grafberger/mlinspect) project as a foundation, mainly for its plan extraction from native ML pipelines.

This demo app is built using [Streamlit](https://streamlit.io).

Requirements
---

Python 3.9

Usage
---

```shell
# Create a virtual environment
python3.9 -m venv venv
source venv/bin/activate
pip install -U pip

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Visit <http://localhost:8501> in your browser.

<!-- TODO: Caching -->
<!-- TODO: Pages -->
<!-- TODO: Docker -->
<!-- TODO: Deployment -->

License
---

This library is licensed under the Apache 2.0 License.
