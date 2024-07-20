import os
import tarfile

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.s3 import S3Downloader

from .dataset import build_dataset_and_upload

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName=os.environ["IAM_ROLE"])["Role"]["Arn"]

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

training_input_path = f"s3://{sagemaker_session_bucket}/lss/train"
test_input_path = f"s3://{sagemaker_session_bucket}/lss/test"
build_dataset_and_upload(training_input_path, test_input_path)

hyperparameters = {
    "epochs": 3,
    "per_device_train_batch_size": 32,
    "model_name_or_path": "distilbert/distilbert-base-uncased",
}


huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="/lss/jobs/distilbert",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    hyperparameters=hyperparameters,
)

huggingface_estimator.fit({"train": training_input_path, "test": test_input_path})

S3Downloader.download(
    s3_uri=huggingface_estimator.model_data,
    local_path="/data/models/distilbert",
)

with tarfile.open("/data/models/distilbert/model.tar.gz", "r") as tar:
    tar.extractall(path="/data/models/distilbert")
