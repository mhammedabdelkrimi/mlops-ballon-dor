import botocore.session
from botocore.config import Config
from botocore.exceptions import ClientError

s3 = botocore.session.get_session().create_client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id="local",
    aws_secret_access_key="local-secret",
    region_name="us-east-1",
    config=Config(s3={"addressing_style": "path"}),
)

try:
    s3.create_bucket(Bucket="mlops-bucket")
    print("create_bucket : OK")
except ClientError as e:
    print("create_bucket :", e.response["Error"]["Code"])

try:
    s3.head_bucket(Bucket="mlops-bucket")
    print("head_bucket   : le bucket existe")
except ClientError as e:
    print("head_bucket   : ERREUR", e.response["Error"]["Code"])

s3.put_object(Bucket="mlops-bucket", Key="test.txt", Body=b"bonjour")
lu = s3.get_object(Bucket="mlops-bucket", Key="test.txt")["Body"].read()
print("écriture/lecture :", lu)