"""API manager"""

import json
from collections import defaultdict
from datetime import datetime

import pandas as pd
import requests
from loguru import logger


class APIManager:
    def __init__(self, secrets_manager) -> None:
        self.api_v3_key: str = secrets_manager.get_lima_v3_api_key()
        self.api_v3_url: str = secrets_manager.get_lima_v3_api_url()

    def get_fields(self, customer_id):
        response: requests.Response = requests.get(
            url=f"{self.api_v3_url}/fields?farm_id={customer_id}&skip=0&limit=1000",
            params={"customer_id": customer_id},
            headers={"x-api-key": self.api_v3_key},
            timeout=60,
        )
        try:
            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data = json.loads(response.content.decode("utf-8"))

            if not data:
                return []

            results = []

            for item in data:
                # Skip any strings or wrong formats
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in API response: {item}")
                    continue

                results.append(
                    {
                        "field": item.get("name"),
                        "field_id": item.get("id"),
                        "variety_id": item.get("variety_id"),
                        "avg_img_bed_length": item.get("avg_img_bed_length"),
                        "area_msqr": item.get("area_msqr"),
                        "image_bed_number": item.get("image_bed_number"),
                        "location": item.get("location"),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Unexpected error in get_varieties: {e}")
            return []

    def get_varieties(self, farm_id):
        """
        Get varieties
        """
        try:
            response: requests.Response = requests.get(
                url=f"{self.api_v3_url}/varieties",
                params={
                    "skip": 0,
                    "limit": 1000,
                },
                headers={
                    "x-api-key": self.api_v3_key,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data: dict = json.loads(response.content.decode("utf-8"))

            if not data:
                return []

            results = [
                {
                    "farm_id": item["farm_id"],
                    "variety_id": item["id"],
                    "variety_name": item["name"],
                    "type": item["type"],
                }
                for item in data
                if item["farm_id"] == farm_id
            ]

            return results

        except Exception as e:
            logger.error(f"Unexpected error in get_varieties: {e}")
            return []

    def get_forecasts(
        self, variety_id: int, customer_id: int, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get latest forecasts and return as pandas DataFrame in long format

        Returns:
            pd.DataFrame: Long format DataFrame with columns:
                        [variety_name, variety_id, date, total_forecasts, field, field_forecasts]
        """
        try:
            response: requests.Response = requests.get(
                url=f"{self.api_v3_url}/varieties/{variety_id}/latest-forecasts",
                params={
                    "customer_id": customer_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "granularity": "weeks",
                    "all_farms": "false",
                },
                headers={
                    "x-api-key": self.api_v3_key,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data: dict = json.loads(response.content.decode("utf-8"))

            if not data:
                return pd.DataFrame()

            # Extract basic information
            variety_name = data["variety_name"]
            variety_id_from_response = data["query_metadata"].get(
                "variety_id", variety_id
            )
            dates = pd.to_datetime(data["dates"])
            total_forecasts = data["forecasts"]

            # Create list to store all rows
            rows = []

            # Create rows for total forecasts
            for date, forecast in zip(dates, total_forecasts):
                rows.append(
                    {
                        "variety_name": variety_name,
                        "variety_id": variety_id_from_response,
                        "date": date,
                        "field": "TOTAL",
                        "field_forecasts": forecast,
                    }
                )

            # Create rows for each field
            if "field_forecasts" in data and data["field_forecasts"]:
                for field_data in data["field_forecasts"]:
                    for field_name, field_info in field_data.items():
                        if "forecasts" in field_info:
                            field_forecasts = field_info["forecasts"]

                            for date, field_forecast in zip(dates, field_forecasts):
                                rows.append(
                                    {
                                        "variety_name": variety_name,
                                        "variety_id": variety_id_from_response,
                                        "date": date,
                                        "field": field_name,
                                        "field_forecasts": field_forecast,
                                    }
                                )

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Sort by date and field for better organization
            if not df.empty:
                df = df.sort_values(["date", "field"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Unexpected error in get_forecasts_long_format: {e}")
            return pd.DataFrame()

    def get_raw_forecasts(
        self, variety_id: int, customer_id: int, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get latest raw forecasts and return as pandas DataFrame in long format

        Returns:
            pd.DataFrame: Long format DataFrame with columns:
                        [variety_name, variety_id, date, total_forecasts, field, field_forecasts]
        """
        try:
            response: requests.Response = requests.get(
                url=f"{self.api_v3_url}/varieties/{variety_id}/latest-forecasts",
                params={
                    "customer_id": customer_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "granularity": "weeks",
                    "all_farms": "false",
                    "raw_forecast": "true",
                },
                headers={
                    "x-api-key": self.api_v3_key,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data: dict = json.loads(response.content.decode("utf-8"))

            if not data:
                return pd.DataFrame()

            # Extract basic information
            variety_name = data["variety_name"]
            variety_id_from_response = data["query_metadata"].get(
                "variety_id", variety_id
            )
            dates = pd.to_datetime(data["dates"])
            total_forecasts = data["forecasts"]

            # Create list to store all rows
            rows = []

            # Create rows for total forecasts
            for date, forecast in zip(dates, total_forecasts):
                rows.append(
                    {
                        "variety_name": variety_name,
                        "variety_id": variety_id_from_response,
                        "date": date,
                        "field": "TOTAL",
                        "field_raw_forecasts": forecast,
                    }
                )

            # Create rows for each field
            if "field_forecasts" in data and data["field_forecasts"]:
                for field_data in data["field_forecasts"]:
                    for field_name, field_info in field_data.items():
                        if "forecasts" in field_info:
                            field_forecasts = field_info["forecasts"]

                            for date, field_forecast in zip(dates, field_forecasts):
                                rows.append(
                                    {
                                        "variety_name": variety_name,
                                        "variety_id": variety_id_from_response,
                                        "date": date,
                                        "field": field_name,
                                        "field_raw_forecasts": field_forecast,
                                    }
                                )

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Sort by date and field for better organization
            if not df.empty:
                df = df.sort_values(["date", "field"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Unexpected error in get_forecasts_long_format: {e}")
            return pd.DataFrame()

    def get_harvests(
        self, variety_id: int, customer_id: int, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Get latest forecasts and return as pandas DataFrame in long format

        Returns:
            pd.DataFrame: Long format DataFrame with columns:
                        [variety_name, variety_id, date, total_forecasts, field, field_forecasts]
        """
        try:
            response: requests.Response = requests.get(
                url=f"{self.api_v3_url}/varieties/{variety_id}/graph-harvests",
                params={
                    "customer_id": customer_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "granularity": "weeks",
                    "all_farms": "false",
                },
                headers={
                    "x-api-key": self.api_v3_key,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data: dict = json.loads(response.content.decode("utf-8"))

            if not data:
                return pd.DataFrame()

            # Extract basic information
            variety_name = data["variety_name"]
            variety_id = data["query_metadata"]["variety_id"]
            dates = pd.to_datetime(data["dates"])

            # Create list to store all rows
            rows = []
            if "field_harvests" in data and data["field_harvests"]:
                for field_data in data["field_harvests"]:
                    for field_name, field_info in field_data.items():
                        if "harvests" in field_info:
                            field_harvests = field_info["harvests"]

                            for date, field_harvest in zip(dates, field_harvests):
                                rows.append(
                                    {
                                        "variety_name": variety_name,
                                        "variety_id": variety_id,
                                        "date": date,
                                        "field": field_name,
                                        "field_harvests": field_harvest,
                                    }
                                )

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Sort by date and field for better organization
            if not df.empty:
                df = df.sort_values(["date", "field"]).reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Unexpected error in get_forecasts_long_format: {e}")
            return pd.DataFrame()

    def get_grade_map(self) -> dict:
        """
        Build a mapping from production CATEGORY ID to grade column name.

        Returns:
            {
                10: "grade_35cm",
                11: "grade_40cm",
                ...
                19: "rejects"
            }
        """

        response = requests.get(
            url=f"{self.api_v3_url}/production-categories",
            headers={"x-api-key": self.api_v3_key},
            timeout=30,
        )

        if response.status_code != 200:
            logger.warning(
                f"Failed to fetch production categories "
                f"(status={response.status_code})"
            )
            return {}

        categories = response.json()
        if not categories:
            logger.warning("Production categories endpoint returned empty list")
            return {}

        grade_map = {}

        for cat in categories:
            cat_id = cat.get("id")
            name = cat.get("name")
            flags = cat.get("flags", "")

            if not cat_id or not name:
                continue

            if "grade" in flags:
                grade_map[cat_id] = name
            elif "reject" in flags:
                grade_map[cat_id] = "rejects"

        if not grade_map:
            logger.warning("No grade or reject categories found")

        return grade_map

    def get_harvests_by_grade_wide(
        self,
        variety_id: int,
        customer_id: int,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:

        grade_map = self.get_grade_map()

        response = requests.get(
            url=f"{self.api_v3_url}/varieties/{variety_id}/graph-harvests",
            params={
                "customer_id": customer_id,
                "start_date": start_date,
                "end_date": end_date,
                "granularity": "weeks",
                "all_farms": "false",
            },
            headers={"x-api-key": self.api_v3_key},
            timeout=60,
        )

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        if not data:
            return pd.DataFrame()

        rows = []
        dates = pd.to_datetime(data["dates"])

        def explode(detailed_views, field_name):
            for date, week_view in zip(dates, detailed_views):
                row = {
                    "variety_name": data["variety_name"],
                    "variety_id": variety_id,
                    "date": date,
                    "field": field_name,
                }

                for cat_id, value in week_view.items():
                    col = grade_map.get(int(cat_id))
                    if col:
                        row[col] = value

                rows.append(row)

        # TOTAL
        explode(data["detailed_view"], "TOTAL")

        # FIELD level
        for field_block in data.get("field_harvests", []):
            for field_name, field_info in field_block.items():
                explode(field_info["detailed_view"], field_name)

        df = pd.DataFrame(rows)

        # Ensure all grade columns exist
        for col in grade_map.values():
            if col not in df.columns:
                df[col] = 0

        grade_cols = sorted(grade_map.values())

        return (
            df[["variety_name", "variety_id", "date", "field"] + grade_cols]
            .fillna(0)
            .sort_values(["date", "field"])
            .reset_index(drop=True)
        )

    def get_forecasts_by_grade_wide(
        self,
        variety_id: int,
        customer_id: int,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get weekly forecasts split by production grades (WIDE format)

        Returns:
            variety_name | variety_id | date | field | grade_XX | ... | rejects
        """

        try:
            # Fetch grade definitions (VARIETY-SPECIFIC)
            grade_map = self.get_grade_map()

            response = requests.get(
                url=f"{self.api_v3_url}/varieties/{variety_id}/latest-forecasts",
                params={
                    "customer_id": customer_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "granularity": "weeks",
                    "all_farms": "false",
                },
                headers={"x-api-key": self.api_v3_key},
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data = response.json()
            if not data:
                return pd.DataFrame()

            variety_name = data["variety_name"]
            variety_id_from_response = data["query_metadata"].get(
                "variety_id", variety_id
            )
            dates = pd.to_datetime(data["dates"])

            rows = []

            def explode(detailed_views, field_name):
                for date, week_view in zip(dates, detailed_views):
                    row = {
                        "variety_name": variety_name,
                        "variety_id": variety_id_from_response,
                        "date": date,
                        "field": field_name,
                    }

                    for cat_id, value in week_view.items():
                        col = grade_map.get(int(cat_id))
                        if col:
                            row[col] = value

                    rows.append(row)


            if "detailed_view" in data:
                explode(data["detailed_view"], "TOTAL")

            for field_block in data.get("field_forecasts", []):
                for field_name, field_info in field_block.items():
                    if "detailed_view" in field_info:
                        explode(field_info["detailed_view"], field_name)

            df = pd.DataFrame(rows)

            # Ensure all grade columns exist
            for col in grade_map.values():
                if col not in df.columns:
                    df[col] = 0

            grade_cols = sorted(grade_map.values())

            return (
                df[["variety_name", "variety_id", "date", "field"] + grade_cols]
                .fillna(0)
                .sort_values(["date", "field"])
                .reset_index(drop=True)
            )

        except Exception as e:
            logger.error(f"Unexpected error in get_forecasts_by_grade_wide: {e}")
            return pd.DataFrame()

    def get_raw_forecasts_by_grade_wide(
        self,
        variety_id: int,
        customer_id: int,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get weekly RAW forecasts split by production grades (WIDE format)

        Returns:
            variety_name | variety_id | date | field | grade_XX | ... | rejects
        """

        try:
            # Fetch grade definitions (VARIETY-SPECIFIC)
            grade_map = self.get_grade_map()

            response = requests.get(
                url=f"{self.api_v3_url}/varieties/{variety_id}/latest-forecasts",
                params={
                    "customer_id": customer_id,
                    "start_date": start_date,
                    "end_date": end_date,
                    "granularity": "weeks",
                    "all_farms": "false",
                    "raw_forecast": "true",
                },
                headers={"x-api-key": self.api_v3_key},
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data = response.json()
            if not data:
                return pd.DataFrame()

            variety_name = data["variety_name"]
            variety_id_from_response = data["query_metadata"].get(
                "variety_id", variety_id
            )
            dates = pd.to_datetime(data["dates"])

            rows = []

            def explode(detailed_views, field_name):
                for date, week_view in zip(dates, detailed_views):
                    row = {
                        "variety_name": variety_name,
                        "variety_id": variety_id_from_response,
                        "date": date,
                        "field": field_name,
                    }

                    for cat_id, value in week_view.items():
                        col = grade_map.get(int(cat_id))
                        if col:
                            row[col] = value

                    rows.append(row)


            if "detailed_view" in data:
                explode(data["detailed_view"], "TOTAL")

            for field_block in data.get("field_forecasts", []):
                for field_name, field_info in field_block.items():
                    if "detailed_view" in field_info:
                        explode(field_info["detailed_view"], field_name)

            df = pd.DataFrame(rows)

            # Ensure all grade columns exist
            for col in grade_map.values():
                if col not in df.columns:
                    df[col] = 0

            grade_cols = sorted(grade_map.values())

            return (
                df[["variety_name", "variety_id", "date", "field"] + grade_cols]
                .fillna(0)
                .sort_values(["date", "field"])
                .reset_index(drop=True)
            )

        except Exception as e:
            logger.error(f"Unexpected error in get_raw_forecasts_by_grade_wide: {e}")
            return pd.DataFrame()

    def get_stage_durations(self, variety_id):
        response: requests.Response = requests.get(
            url=f"{self.api_v3_url}/growth-stages/associations/by-variety?variety_id={variety_id}",
            headers={"x-api-key": self.api_v3_key},
            timeout=60,
        )
        try:
            logger.debug(f"fetching data done variety id - {variety_id}")
            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data = json.loads(response.content.decode("utf-8"))

            if not data:
                return []

            results = []

            for item in data:
                # Skip any strings or wrong formats
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item in API response: {item}")
                    continue

                results.append(
                    {
                        "farm_id": item.get("farm_id"),
                        "id": item.get("growth_stage_id"),
                        "variety_id": item.get("growth_stage", {})["variety_id"],
                        "name": item.get("growth_stage", {})["name"],
                        "duration": item.get("duration"),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Unexpected error in get_varieties: {e}")
            return []

    def get_raw_forecasts_series(self, field_id: int, entry_date: str) -> pd.DataFrame:
        """
        Get raw forecasts series
        """
        try:
            response: requests.Response = requests.get(
                url=f"{self.api_v3_url}/forecasts/{field_id}/{entry_date}?",
                params={
                    "allow_all": "true",
                    "raw_forecast": "true",
                },
                headers={
                    "x-api-key": self.api_v3_key,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data: dict = json.loads(response.content.decode("utf-8"))

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["created_at"]).dt.date


            agg = df.groupby(
                ["field_id", "forecast_serie_id", "entry_date"], as_index=False
            )["entry_value"].sum()

            result = (
                agg.pivot(
                    index=["entry_date", "field_id"],
                    columns="forecast_serie_id",
                    values="entry_value",
                )
                .fillna(0)
                .reset_index()
            )

            result.columns = [
                str(c) if isinstance(c, (int, float)) else c for c in result.columns
            ]

            return result

        except Exception as e:
            logger.error(f"Unexpected error in get_raw_forecasts_series: {e}")
            return pd.DataFrame()

    def get_final_forecasts_series(
        self, field_id: int, entry_date: str
    ) -> pd.DataFrame:
        """
        Get raw forecasts series
        """

        try:
            response: requests.Response = requests.get(
                url=f"{self.api_v3_url}/forecasts/{field_id}/{entry_date}?",
                params={
                    "allow_all": "true",
                    "raw_forecast": "false",
                },
                headers={
                    "x-api-key": self.api_v3_key,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"API request failed with status {response.status_code}: {response.content}"
                )
                return pd.DataFrame()

            data: dict = json.loads(response.content.decode("utf-8"))

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            try:
                df["date"] = pd.to_datetime(df["created_at"]).dt.date
            except:
                df["date"] = pd.to_datetime(df["created_at"], format="mixed").dt.date

            agg = df.groupby(
                ["field_id", "forecast_serie_id", "entry_date"], as_index=False
            )["entry_value"].sum()

            result = (
                agg.pivot(
                    index=["entry_date", "field_id"],
                    columns="forecast_serie_id",
                    values="entry_value",
                )
                .fillna(0)
                .reset_index()
            )

            result.columns = [
                str(c) if isinstance(c, (int, float)) else c for c in result.columns
            ]

            return result

        except Exception as e:
            logger.error(f"Unexpected error in get_final_forecasts_series: {e}")
            return pd.DataFrame()
