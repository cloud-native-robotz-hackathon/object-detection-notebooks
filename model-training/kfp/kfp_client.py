"""
KFP Client helper for OpenShift AI / Data Science Pipelines.

Provides a convenience class that compiles a KFP pipeline, uploads it
to the DSP API server running in the same OpenShift namespace, and
optionally starts a pipeline run.

Adapted from
https://github.com/mamurak/os-mlops/blob/main/notebooks/fraud-detection-onnx/training/sdk/kfp_client.py

Usage
-----
::

    from kfp_pipeline import model_training_pipeline
    from kfp_client import KfpPipeline

    pipeline = KfpPipeline(
        model_training_pipeline,
        'object-detection-training',
    )
    pipeline.run_with_parameters(
        pipeline_parameters={
            'batch_size': 16,
            'epochs': 300,
            'base_model': 'yolov5n',
            'model_object_prefix': 'model',
        },
        experiment_name='object-detection',
    )
"""

from datetime import datetime

from kfp.client import Client
from kfp.compiler import Compiler

_compiler = Compiler()


class KfpPipeline:
    """Compile, upload, and run a KFP pipeline on Data Science Pipelines."""

    def __init__(self, sdk_pipeline, pipeline_name, caching=False):
        self.pipeline_name = pipeline_name
        self.caching = caching
        self._client = _get_kfp_client()
        self._pipeline_id = None
        self._pipeline_version_id = None
        self._compiled_pipeline_path = './pipeline.yaml'
        self._upload_pipeline(sdk_pipeline)

    # ── internal helpers ──────────────────────────────────────────

    def _upload_pipeline(self, sdk_pipeline):
        print('Compiling pipeline …')
        _compiler.compile(
            sdk_pipeline,
            package_path=self._compiled_pipeline_path,
        )
        print(f'Compiled pipeline to {self._compiled_pipeline_path}')

        print(f'Uploading pipeline "{self.pipeline_name}" …')
        pipeline_id = self._client.get_pipeline_id(self.pipeline_name)

        if not pipeline_id:
            print(
                f'Pipeline "{self.pipeline_name}" not found – creating …'
            )
            remote_pipeline = self._client.upload_pipeline(
                pipeline_package_path=self._compiled_pipeline_path,
                pipeline_name=self.pipeline_name,
                description=(
                    'YOLOv5 object-detection training pipeline '
                    '(deployed via KFP SDK)'
                ),
            )
            pipeline_id = remote_pipeline.pipeline_id

        self._pipeline_id = pipeline_id

        version_name = f'model-training-{_timestamp()}'
        print(
            f'Uploading new version "{version_name}" for pipeline '
            f'{pipeline_id} …'
        )
        remote_version = self._client.upload_pipeline_version(
            pipeline_package_path=self._compiled_pipeline_path,
            pipeline_version_name=version_name,
            pipeline_name=self.pipeline_name,
        )
        self._pipeline_version_id = remote_version.pipeline_version_id
        print(
            f'Uploaded version "{version_name}" '
            f'(ID {self._pipeline_version_id})'
        )

    # ── public API ────────────────────────────────────────────────

    def run_with_parameters(self, pipeline_parameters, experiment_name):
        """Start a pipeline run inside the given experiment."""
        print(f'Starting pipeline run in experiment "{experiment_name}" …')

        try:
            remote_experiment = self._client.get_experiment(
                experiment_name=experiment_name,
            )
        except Exception:
            print(
                f'Experiment "{experiment_name}" not found – creating …'
            )
            remote_experiment = self._client.create_experiment(
                experiment_name,
            )

        experiment_id = remote_experiment.experiment_id
        run_name = f'model-training-run-{_timestamp()}'
        print(f'Submitting run "{run_name}" …')

        self._client.run_pipeline(
            experiment_id=experiment_id,
            job_name=run_name,
            pipeline_id=self._pipeline_id,
            version_id=self._pipeline_version_id,
            params=pipeline_parameters,
            enable_caching=self.caching,
        )
        print('Pipeline run submitted.')


# ── module-private helpers ────────────────────────────────────────

def _get_kfp_client():
    """
    Build a KFP ``Client`` that talks to the DSP API server in the
    current OpenShift namespace, using the service-account token
    mounted into every pod.
    """
    namespace_path = (
        '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    )
    with open(namespace_path, 'r') as fh:
        namespace = fh.read()

    kubeflow_endpoint = (
        f'https://ds-pipeline-dspa.{namespace}.svc:8443'
    )

    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    with open(token_path, 'r') as fh:
        bearer_token = fh.read()

    ssl_ca_cert = (
        '/var/run/secrets/kubernetes.io/serviceaccount/service-ca.crt'
    )

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    return Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
        ssl_ca_cert=ssl_ca_cert,
    )


def _timestamp():
    return datetime.now().strftime('%y%m%d%H%M')
