import pandas as pd
import numpy as np
import torch
import parameters as param
import logging
import pickle
import datetime
import time
import argparse
import yaml
import os
from googleapiclient import discovery
import call_training as call_training
from preprocessing import preprocessing
from model import CustomLSTM
from attribution_model import attribution_model
import utilities as custom_util
import sys

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trigger an attribution comportementale.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", dest="mode", help="mode switch : train or evaluate")
    parser.add_argument("--conf", dest="config", default="../../conf/infrastructure.yaml",
                        help="absolute or relative path of configuration file")
    parser.add_argument("--env", dest="env", help="environment : local or cloud", default="local")
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    root_logger = logging.getLogger()
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    # Setting up gcp parameters

    project_global, dataset_ref, bucket, credentials = custom_util.gcp_setting(config)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(dir_path)
    logger.info('Checking if packages needed by ML engine are available')
    packages = {p: 'gs://{bucket}/att_comportementale/packages/{package}'.format(bucket=bucket.name,
                                                                                 package=p
                                                                                 ) for p in os.listdir(
        os.path.join(parent_path, 'packages'))}
    logger.debug("package URIs: %s " % list(packages.values()))
    custom_util.load_package(packages, bucket, parent_path)

    # Setting the random seed
    np.random.seed(param.random_seed)
    if args.mode == 'train':
        if args.env == 'local':
            call_training.training(project_global, bucket.name)
        elif args.env == 'cloud':
            # Training on cloud
            # Size of machine we will do RNN on. Can be S, M or L
            mlmachine_size = 'S'
            logger.info('Start training on the cloud')
            ml_engine_service = discovery.build('ml', 'v1', credentials=credentials)
            job_parent = "projects/{project}".format(project=project_global)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            job_id = "job_{}".format(now_str)
            job_body = {'trainingInput':
                            {'pythonVersion': param.ml_pythonVersion,
                             'runtimeVersion': param.ml_runtimeVersion,
                             'scaleTier': param.typology_machine[mlmachine_size]['ml_scaleTier_train'],
                             'region': param.ml_region,
                             'pythonModule': 'lstm.call_training',
                             'args': ["--global_proj_id", project_global,
                                      "--bucket_name", bucket.name],
                             'packageUris': sorted(list(packages.values())),
                             'masterType': param.typology_machine[mlmachine_size]['ml_masterType'],
                             'workerType': param.typology_machine[mlmachine_size]['ml_workerType'],
                             'workerCount': param.typology_machine[mlmachine_size]['ml_workerCount'],
                             'parameterServerCount': param.typology_machine[mlmachine_size][
                                 'ml_parameterServerCount'],
                             'parameterServerType': param.typology_machine[mlmachine_size][
                                 'ml_parameterServerType']
                             },
                        'jobId': job_id}
            logger.info("job_body: %s" % job_body)
            logger.info("job_parent: %s" % job_parent)
            logger.info("creating a job ml: %s" % job_id)
            job_ml = ml_engine_service.projects().jobs().create(parent=job_parent, body=job_body).execute()
            time.sleep(5)
            try:
                succeeded_job = custom_util.is_success(ml_engine_service, project_global, job_id)
                if succeeded_job:
                    logger.info('Training job done')
                else:
                    logger.error('Training job failed')
                    sys.exit(1)
            except Exception as e:
                logger.error(e)
                sys.exit(1)
            logger.info("[FINISHED] attribution comportementale for account (job: %s)" % args.mode)
        else:
            logger.error('Only local or cloud environment are available')
            sys.exit(1)
    else:
        # Applying the model for evaluating the attribution weights
        logger.info('Loading the model weights')
        model_checkpoint = torch.load(param.model_folder)
        logger.info('Creating the model')
        rnn = CustomLSTM(model_checkpoint['hidden_dimension'], model_checkpoint['vocabulary_size'],
                         model_checkpoint['embedding_dimension'], model_checkpoint['output_size'])
        logger.info('Loading the model weights')
        rnn.load_state_dict(model_checkpoint['state_dict'])
        logger.info('Loading the dataset')
        df = pd.read_csv(param.dataset_name)
        logger.info('Preprocessing')
        df = preprocessing(df, True)
        logger.info('Loading the encoder')
        with open(param.encoder_name, 'rb') as load_encoder:
            encoder = pickle.load(load_encoder)
        df_results = attribution_model(df, encoder, param.attribution_model, rnn)
        if param.dump_attribution_results:
            df_results.to_csv(param.saving_folder + 'attribution_results_' + str(param.attribution_model) + '.csv')
