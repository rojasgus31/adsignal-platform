"""Model training, evaluation and monitoring.

`features.py` is pure domain logic over DataFrames. The rest wraps scikit-learn
models and MLflow. No module here opens a database connection.
"""
