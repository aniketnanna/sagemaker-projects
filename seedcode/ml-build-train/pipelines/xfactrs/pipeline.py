import boto3
import json
import logging
import os
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
import sagemaker.session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.estimator import Estimator
import traceback

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client.

       Args:
           region: the aws region to start the session
           default_bucket: the bucket to use for storing the artifacts

       Returns:
           `sagemaker.session.Session instance
       """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
        region,
        sagemaker_project_arn=None,
        role=None,
        default_bucket=None,
        model_package_group_name="xfactrsPackageGroup",
        pipeline_name="xfactrsPipeline",
        base_job_prefix="xfactrs",
        processing_instance_type="ml.t3.large",
        training_instance_type="ml.m5.large",
        inference_instance_type="ml.m5.large"
):
    pipeline_session = get_pipeline_session(region, default_bucket)

    if role is None:
        role = sagemaker.session.get_execution_role(pipeline_session)

    training_hyperparameters = {
        "max_depth":5,
        "eta":0.2,
        "gamma":4,
        "min_child_weight":6,
        "subsample":0.8,
        "verbosity":0,
        "objective":"binary:logistic",
        "num_round":1,
    }

    input_data = ParameterString(
        name="InputData", default_value="s3://{}/datasets/tabular/customer_churn".format(default_bucket)
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    processing_instance_count_param = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )

    training_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount", default_value=1
    )

    processor = FrameworkProcessor(
        estimator_cls=XGBoost,
        framework_version="1.7-1",
        role=role,
        instance_count=processing_instance_count_param,
        instance_type=processing_instance_type,
        sagemaker_session=pipeline_session
    )

    run_args = processor.get_run_args(
        code=os.path.join(BASE_DIR, "processing.py"),
        inputs=[
            ProcessingInput(
                input_name="input",
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination="s3://{}/data/output/train".format(default_bucket)),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
                destination="s3://{}/data/output/validation".format(default_bucket)),
            ProcessingOutput(
                output_name="inference",
                source="/opt/ml/processing/output/inference",
                destination="s3://{}/inference/data/input".format(default_bucket))
        ]
    )

    step_process = ProcessingStep(
        name="ProcessData",
        code=run_args.code,
        processor=processor,
        inputs=run_args.inputs,
        outputs=run_args.outputs
    )
    
    # Debugging
    # print("BASE_DIR: ", BASE_DIR)
    # print("entry Point: ")
    # for path, subdirs, files in os.walk(BASE_DIR):
    #     for name in files:
    #         print(os.path.join(path, name))
    
    # print("train_model.py path: ", os.path.join(BASE_DIR, "train_model.py"))
    # print("source_dir: ", os.path.join(BASE_DIR, "requirements.txt"))
    # print("output_path: ", "s3://{}/models".format(default_bucket))
    
    image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1",
    # py_version="py3",
    instance_type=training_instance_type
    )
    
    #========|Training with fit method|==================#
    
    # xgb_train = Estimator(
    #     image_uri=image_uri,
    #     instance_type=training_instance_type,
    #     instance_count=training_instance_count_param,
    #     output_path="s3://{}/models".format(default_bucket),
    #     sagemaker_session=pipeline_session,
    #     role=role,
    # )
    
    # xgb_train.set_hyperparameters(
    # max_depth=5,
    # eta=0.2,
    # gamma=4,
    # min_child_weight=6,
    # subsample=0.8,
    # verbosity=0,
    # objective="binary:logistic",
    # num_round=100,
    # )
    
    
    # step_args = xgb_train.fit(
    # inputs={
    #     "train": TrainingInput(
    #         s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
    #         content_type="text/csv"
    #     ),
    #     "validation": TrainingInput(
    #         s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
    #         content_type="text/csv"
    #     )
    # }
    # )

    # step_train = TrainingStep(
    #     depends_on=[step_process],
    #     name="TrainModel",
    #     step_args=step_args,
    # )
    
    #========|Training with Script Mode|==================#
    
    estimator = XGBoost(
         image_uri=image_uri,
         entry_point=os.path.join(BASE_DIR, "train_module-1.py"),
         framework_version="1.7-1",
         py_version="py3",
         output_path="s3://{}/models".format(default_bucket),
         hyperparameters=training_hyperparameters,
         role=role,
         instance_count=training_instance_count_param,
         instance_type=training_instance_type,
         disable_profiler=True
     )
     
     
    step_train = TrainingStep(
         depends_on=[step_process],
         name="TrainModel",
         estimator=estimator,
         inputs={
             "train": TrainingInput(
                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                 content_type="text/csv"
                 ),
                 "validation": TrainingInput(
                     s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                     content_type="text/csv"
                     )
        }
    )
    
    

    # estimator_2 = PyTorch(
    #     os.path.join(BASE_DIR, "train_model_2.py"),
    #     framework_version="1.12",
    #     py_version="py38",
    #     output_path="s3://{}/models".format(default_bucket),
    #     hyperparameters=training_hyperparameters,
    #     role=role,
    #     instance_count=training_instance_count_param,
    #     instance_type=training_instance_type,
    #     disable_profiler=True
    # )

    # step_train_2 = TrainingStep(
    #     depends_on=[step_process],
    #     name="TrainModel2",
    #     estimator=estimator_2,
    #     inputs={
    #         "train": TrainingInput(
    #             s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
    #             content_type="text/csv"
    #         ),
    #         "test": TrainingInput(
    #             s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    #             content_type="text/csv"
    #         )
    #     }
    # )

    step_register_model = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=[inference_instance_type],
        transform_instances=[inference_instance_type]
    )

    # step_register_model_2 = RegisterModel(
    #     name="RegisterModel2",
    #     estimator=estimator_2,
    #     model_data=step_train_2.properties.ModelArtifacts.S3ModelArtifacts,
    #     model_package_group_name=model_package_group_name_2,
    #     approval_status=model_approval_status,
    #     content_types=["text/csv"],
    #     response_types=["text/csv"],
    #     inference_instances=[inference_instance_type],
    #     transform_instances=[inference_instance_type]
    # )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            model_approval_status,
            processing_instance_count_param,
            training_instance_count_param
        ],
        steps=[
            step_process,
            step_train,
            step_register_model,
        ],
        sagemaker_session=pipeline_session
    )

    return pipeline
