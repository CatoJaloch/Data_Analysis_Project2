
import json
from typing import Any

from boto3.session import Session


class SecretsManager:
    def __init__(
        self,
        env: str = "",
    ) -> None:
        self.env: str = env.lower()
        session = Session()
        region_name: str = "eu-west-1"
        self.client: Any = session.client(
            service_name="secretsmanager", region_name=region_name
        )

    def get_secrets(self, secret_name: str) -> dict:
        secrets_result: dict = {}
        secret_value: dict = self.client.get_secret_value(SecretId=secret_name)
        secret_value_dict: dict = json.loads(secret_value["SecretString"])
        for k, v in secret_value_dict.items():
            secrets_result.update({k: v})
        return secrets_result

    def get_lima_x_api_key(self) -> str:
        secret_name: str = f"LIMA_API_X_API_KEY_{self.env}"
        secret_value: str = self.get_secrets(secret_name=secret_name)[
            "LIMA_API_X_API_KEY"
        ]
        return secret_value

    def get_lima_v3_api_key(self) -> str:
        secret_name: str = f"LIMA_API_v3_API_KEY_{self.env.upper()}"
        secret_value: str = self.get_secrets(secret_name=secret_name)[
            "LIMA_API_v3_API_KEY"
        ]
        return secret_value

    def get_lima_v3_api_url(self) -> str:
        secret_name: str = f"LIMA_API_v3_API_KEY_{self.env.upper()}"
        return self.get_secrets(secret_name=secret_name)["LIMA_API_v3_API_URL"]

    def get_lima_api_url(self) -> str:
        secret_name: str = f"LIMA_API_URL_{self.env}"
        return self.get_secrets(secret_name=secret_name)["LIMA_API_URL"]

    def get_series_db_creds(self) -> dict:
        secret_name: str = "LIMA_RETOOL_PROD_DB_SECRET"
        creds = self.get_secrets(secret_name=secret_name)
        secret_name: str = "series_db_metadata"
        metadata = self.get_secrets(secret_name=secret_name)
        results = {
            "db_user": creds["username"],
            "db_pwd": creds["password"],
            "db_name": metadata["DB_NAME"],
            "db_port": metadata["DB_PORT"],
            "db_host": metadata["DB_HOST"],
        }
        return results
