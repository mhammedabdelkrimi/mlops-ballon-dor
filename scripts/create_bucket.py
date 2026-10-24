import os
import time
import botocore.session
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
s3 = botocore.session.get_session().create_client(
    "s3", endpoint_url=ENDPOINT, aws_access_key_id="local",
    aws_secret_access_key="local-secret", region_name="us-east-1",
    config=Config(s3={"addressing_style": "path"}),
)
for _ in range(30):
    try:
        s3.create_bucket(Bucket="mlops-bucket")
        print("Bucket créé")
        break
    except ClientError as e:
        if e.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print("Bucket déjà présent")
            break
        raise
    except Exception as e:
        print("En attente du stockage...", type(e).__name__)
        time.sleep(2)
else:
    raise SystemExit("Stockage injoignable")