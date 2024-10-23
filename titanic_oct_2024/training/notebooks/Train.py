# Databricks notebook source
##################################################################################
# Model Training Notebook
#
# This notebook shows an example of a Model Training pipeline using Delta tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``titanic_oct_2024/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the trained model in Unity Catalog. 
#  
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_table",
    "bmacxx.dev.titanic",
    label="Name of training dataset",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-titanic_oct_2024-experiment",
    label="MLflow experiment name",
)
# Unity Catalog registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", "dev.titanic_oct_2024.titanic_oct_2024-model", label="Full (Three-Level) Model Name"
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
input_table_name = dbutils.widgets.get("training_table")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1, Load raw data
training_df = spark.table(input_table_name)
training_df.display()

# COMMAND ----------

# DBTITLE 1, Helper function
from mlflow.tracking import MlflowClient
import mlflow.pyfunc


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM model on the data, then log and register the model with MLflow.

# COMMAND ----------

# DBTITLE 1, Train model
import mlflow
from sklearn.model_selection import train_test_split
import catboost

data = training_df.toPandas()
data = data.dropna()
data.drop(columns=['PassengerId', 'Name'], inplace=True)

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["Survived"], axis=1)
X_test = test.drop(["Survived"], axis=1)
y_train = train.Survived
y_test = test.Survived

from catboost import Pool, CatBoostClassifier
categories = ['Cabin', 'Pclass', 'Sex', 'Embarked', 'Ticket']
titanic_train_pool = Pool(X_train, y_train, cat_features=categories)
titanic_test_pool = Pool(X_test, y_test, cat_features=categories)
model = CatBoostClassifier(custom_loss=['Accuracy'])

with mlflow.start_run() as mlflow_run:
    model.fit(titanic_train_pool, eval_set=titanic_test_pool, early_stopping_rounds=20)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.catboost.log_model(model,artifact_path="catboost_model", input_example = X_train.iloc[[0]], registered_model_name=model_name)


# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
