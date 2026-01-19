import sagemaker
import boto3
from access import BUCKET_NAME_S3, PREFIX

boto_sess = boto3.client('sagemaker', region_name='us-east-1')
sess = sagemaker.Session(boto_session=boto3.Session(region_name='us-east-1'))
region_name = sess.boto_session.region_name



train_path = sess.upload_data(
    path = 'train_data.csv',
    bucket=BUCKET_NAME_S3,
    key_prefix=PREFIX
)

test_path = sess.upload_data(
    path = 'test_data.csv',
    bucket=BUCKET_NAME_S3,
    key_prefix=PREFIX
)

print(f'train: {train_path}')
print(f'test: {test_path}')