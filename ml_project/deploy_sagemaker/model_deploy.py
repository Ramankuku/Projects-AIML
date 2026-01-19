from sagemaker.sklearn.estimator import SKLearn
from access import ROLE, FRAMEWORK, TEST_PATH, TRAIN_PATH, OUTPUT_PATH, MODEL_NAME
from sagemaker.sklearn.model import SKLearnModel

hyperparameters = {
    'n_estimators': 100,
    'criterion':'entropy',
    'max_features': 'sqrt',
    'max_depth': 15,
    'min_samples_split': 2,
}

sklearn = SKLearn(
    entry_point='train.py',
    role = ROLE,
    framework_version=FRAMEWORK,
    instance_count=1,
    train_instance_type="ml.m5.xlarge",
    output_path=OUTPUT_PATH,
    hyperparameters=hyperparameters,
    use_spot_instances=False,
    max_run=3600
)

sklearn.fit({'train': TRAIN_PATH, 'test': TEST_PATH})



model = SKLearnModel(
    name=MODEL_NAME,
    model_data = sklearn.model_data,
    role = ROLE,
    entry_point='train.py',
    framework_version=FRAMEWORK
)
predictor = model.deploy(endpoint_name='rf-prediciton-endpoint-new', instance_type="ml.m5.xlarge", initial_instance_count=1)