from datetime import datetime
from typing import Any

import boto3
from loguru import logger

from src.api.api import APIManager
from src.api.secrets_manager import SecretsManager


class AWSHandler:
    def __init__(self, silent: bool = False, quick_setup: bool = False) -> None:
        self.sm = None
        self.silent = silent
        if quick_setup:
            self.create_secrets_manager()
            self.create_s3_resource()

    def create_secrets_manager(self, env: str = "prod") -> SecretsManager:
        self.sm = SecretsManager(env=env)
        return self.sm

    def create_s3_resource(self) -> Any:
        self.s3 = boto3.resource(
            "s3",
        )
        return self.s3

    def create_v3_api_manager(self, env: str):
        if self.sm is None:
            self.sm = self.create_secrets_manager(env=env)
        self.v3_api = APIManager(secrets_manager=self.sm)
        return self.v3_api
