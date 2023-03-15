import os

from mlwhatif.utils import get_project_root


PIPELINE_CONFIG = {
    "reviews": {
        "filename": os.path.join(str(get_project_root()), "demo", "advanced_features", "reviews.py"),
        "columns": ["total_votes", "star_rating", "vine", "category", "review_body", "review_date"],
    },
    "healthcare": {
        "filename": os.path.join(str(get_project_root()), "demo", "feature_overview", "healthcare.py"),
        "columns": ["smokes", "weight", "gave_consent", "ssn", "notes", "hospital"],
    },
}
