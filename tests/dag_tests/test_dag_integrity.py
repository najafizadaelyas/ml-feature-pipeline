"""DAG integrity tests — verify structure without triggering Spark or S3 calls."""

import os
import sys

import pytest

# Ensure the repo root is on sys.path so `dags` and `src` are importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(scope="module")
def dag_bag():
    """Load DAGs from the dags/ directory using Airflow's DagBag."""
    from airflow.models import DagBag

    dags_folder = os.path.join(REPO_ROOT, "dags")
    bag = DagBag(dag_folder=dags_folder, include_examples=False)
    return bag


def test_no_import_errors(dag_bag):
    assert dag_bag.import_errors == {}, (
        f"DAG import errors: {dag_bag.import_errors}"
    )


def test_feature_pipeline_dag_exists(dag_bag):
    assert "feature_pipeline" in dag_bag.dags, (
        f"Expected 'feature_pipeline' DAG. Found: {list(dag_bag.dags.keys())}"
    )


def test_expected_task_count(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]
    task_ids = dag.task_ids
    assert len(task_ids) == 4, f"Expected 4 tasks, got {len(task_ids)}: {task_ids}"


def test_expected_task_ids(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]
    expected = {"load_config", "ingest_raw", "validate", "compute_features"}
    assert set(dag.task_ids) == expected, f"Unexpected task IDs: {dag.task_ids}"


def test_task_dependency_order(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]

    def upstream_ids(task_id: str) -> set[str]:
        return {t.task_id for t in dag.get_task(task_id).upstream_list}

    assert upstream_ids("load_config") == set()
    assert "load_config" in upstream_ids("ingest_raw")
    assert "ingest_raw" in upstream_ids("validate")
    assert "validate" in upstream_ids("compute_features")


def test_dag_schedule(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]
    assert dag.schedule_interval == "0 2 * * *"


def test_dag_has_no_cycles(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]
    # topological_sort raises CycleError if cycles are detected
    dag.topological_sort()


def test_catchup_is_false(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]
    assert dag.catchup is False


def test_max_active_runs(dag_bag):
    dag = dag_bag.dags["feature_pipeline"]
    assert dag.max_active_runs == 1
