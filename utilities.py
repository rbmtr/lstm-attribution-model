import numpy as np
from google.cloud import bigquery, storage
import datetime
import logging
import time
from google.oauth2 import service_account
import os

logger = logging.getLogger(__name__)


# Defining a confusion matrix function

def confusion_matrix(y_real, y_pred):
    """
    :param y_real: numpy array. The real target 
    :param y_pred: numpy array. The predicted target
    :return: numpy array
    """""
    tn, tp, fn, fp = 0, 0, 0, 0
    for i in range(len(y_real)):
        tn += not (y_real[i] or y_pred[i])
        tp += y_real[i] and y_pred[i]
        fn += y_real[i] and not (y_pred[i])
        fp += not (y_real[i]) and y_pred[i]
    return np.array([[tp, fp], [fn, tn]])


def load_from_storage(data_prefix, project_id, bucket_name, source, destination):
    """
    Transfer several data from storage to local
    :param data_prefix: string, the names of the data to transfer
    :param project_id: str, the name of the project id
    :param bucket_name: str, the name of the bucket
    :param source: str, the path in the bucket where the data are stored
    :param destination: str, the path to the destination to which the data are transferred
    :return:
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    for i in bucket.list_blobs(prefix=source + data_prefix):
        i.download_to_filename(destination + i.name.replace(source, ''))


def push_to_storage(data, project_id, bucket_name, source, destination, timeout=60):
    """
    Transfer data from local to storage
    :param data: list, the names of the data to transfer
    :param project_id: str, the name of the project id
    :param bucket_name: str, the name of the bucket
    :param source: str, the path in the local where the data are stored
    :param destination: str, the path to the storage destination to which the data are transferred
    :param timeout: int, the timeout after which the connection to google storage is stopped
    :return:
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination + data)
    # blob.upload_from_filename(source + data, timeout=timeout)
    blob.upload_from_filename(source + data)


def check_existence(data, project_id, bucket_name, source):
    """
    :param data: str, the name of the data to look in storage
    :param project_id: str, the name of the project id
    :param bucket_name: str, the name of the bucket
    :param source: str, the path to the storage destination where the data are stored
    :return:
    """
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source + data)
    return blob.exists()


def is_success(ml_engine_service, project_id, job_id):
    """
    :param ml_engine_service: discovery.build, the container to run the job
    :param project_id: str, the name of the project id
    :param job_id: str, the name of the job id
    :return:
    """
    # Doc: https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#State

    wait = 60  # seconds
    timeout_preparing = datetime.timedelta(seconds=900)
    timeout_running = datetime.timedelta(hours=24)
    api_call_time = datetime.datetime.now()
    api_job_name = "projects/{project_id}/jobs/{job_name}".format(project_id=project_id, job_name=job_id)
    job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
    while not job_description["state"] in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        # check here the PREPARING and RUNNING state to detect the abnormalities of ML Engine service
        if job_description["state"] == "PREPARING":
            delta = datetime.datetime.now() - api_call_time
            if delta > timeout_preparing:
                logger.error("[ML] PREPARING stage timeout after %ss --> CANCEL job '%s'" % (delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception

        if job_description["state"] == "RUNNING":
            delta = datetime.datetime.now() - api_call_time
            if delta > timeout_running + timeout_preparing:
                logger.error("[ML] RUNNING stage timeout after %ss --> CANCEL job '%s'" % (delta.seconds, job_id))
                ml_engine_service.projects().jobs().cancel(name=api_job_name, body={}).execute()
                raise Exception

        logger.info("[ML] NEXT UPDATE for job '%s' IN %ss (%ss ELAPSED IN %s STAGE)" % (job_id,
                                                                                        wait,
                                                                                        delta.seconds,
                                                                                        job_description["state"]))
        job_description = ml_engine_service.projects().jobs().get(name=api_job_name).execute()
        time.sleep(wait)

    logger.info("Job '%s' done" % job_id)

    # Check the job state

    if job_description["state"] == "SUCCEEDED":
        logger.info("Job '%s' succeeded!" % job_id)
        return True
    else:
        logger.error(job_description["errorMessage"])
        return False


def gcp_setting(config):
    """
    :param config: str, the path to the config file
    :return:
    """
    project_global = config['outputs_location']['project_id']
    if config['gcp']['credentials_json_file'] != "":
        credentials = service_account.Credentials.from_service_account_file(
            config['gcp']['credentials_json_file'])
        gs_client = storage.Client(project=project_global, credentials=credentials)
        bq_client = bigquery.Client(project=project_global, credentials=credentials)
        # bq_client_account = bigquery.Client(project=project_account, credentials=credentials)
    else:
        credentials = None
        gs_client = storage.Client(project=project_global)
        bq_client = bigquery.Client(project=project_global)
    dataset_ref = bq_client.dataset(config['outputs_location']['dataset_id'])
    bucket = gs_client.bucket(config['bucket_name'])
    return project_global, dataset_ref, bucket, credentials


def load_package(packages, bucket, parent_path):
    """
    :param packages: list, the list of packages to load
    :param bucket: bucket, the bucket of the project
    :param parent_path: str, the path to the local project folder
    :return:
    """
    for package_name, uri in packages.items():
        package_uri = uri.strip().split("gs://{bucket}/".format(bucket=bucket.name))[1]
        blob = bucket.blob(package_uri)
        if not blob.exists():
            logger.warning("blob %s does not exist on Google Storage, uploading..." % blob)
            blob.upload_from_filename(os.path.join(parent_path, 'packages', package_name))
            logger.info("blob %s available on Google Storage" % blob)
        else:
            logger.info("blob %s does exist on Google Storage, re-uploading..." % blob)
            blob.delete()
            blob.upload_from_filename(os.path.join(parent_path, 'packages', package_name))
            logger.info("blob %s available on Google Storage" % blob)
