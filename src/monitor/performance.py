import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from src.api.aws import AWSHandler

aws = AWSHandler()
api = aws.create_v3_api_manager("prod")


class HorizonPerformance:

    def __init__(
        self,
        farm_id: int,
        horizon_week: str,
        max_horizon: int = 4,
        forecast_kind: str = "raw",
        output_dir: str = "full_performance_output",
    ):
        self.farm_id = farm_id
        self.horizon_week = horizon_week
        self.entry_date = (
            datetime.strptime(horizon_week + "-1", "%G-W%V-%u") - timedelta(weeks=3)
        ).strftime("%Y-%m-%d")
        self.start_week = datetime.strptime(self.entry_date, "%Y-%m-%d").strftime(
            "%G-W%V"
        )
        self.max_horizon = max_horizon
        self.forecast_kind = forecast_kind
        self.output_dir = output_dir

    @staticmethod
    def _parse_week_year(s: str):
        """Parse week string like '2025-W01' to (year, week)"""
        m = re.match(r"^\s*(\d{4})-W?(\d{1,2})\s*$", str(s))
        if not m:
            raise ValueError(f"Bad week format: {s!r}")
        return (int(m.group(1)), int(m.group(2)))

    @staticmethod
    def _to_week_year(dt_series: pd.Series) -> pd.Series:
        """Convert date to ISO week format YYYY-Www"""
        iso = pd.to_datetime(dt_series).dt.isocalendar()
        return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    def _fetch_forecasts_weekly(self, field_id: int, kind: str) -> pd.DataFrame:
        """Fetch daily forecasts and convert to weekly totals"""
        if kind == "final":
            seried_df = api.get_final_forecasts_series(
                field_id=field_id, entry_date=str(self.entry_date)
            )
        elif kind == "raw":
            seried_df = api.get_raw_forecasts_series(
                field_id=field_id, entry_date=str(self.entry_date)
            )
        else:
            raise ValueError("kind must be 'final' or 'raw'")

        seried_df = pd.DataFrame(seried_df)
        if seried_df.empty:
            return pd.DataFrame()

        seried_df["entry_date"] = pd.to_datetime(seried_df["entry_date"])
        seried_df["week_year"] = self._to_week_year(seried_df["entry_date"])

        value_cols = list(
            seried_df.columns.difference(["entry_date", "field_id", "week_year"])
        )

        weekly = seried_df.groupby(["field_id", "week_year"], as_index=False)[
            value_cols
        ].sum()

        return self._dedupe_forecast_columns_weekly(weekly)

    def _dedupe_forecast_columns_weekly(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate forecast series covering same weeks, keep latest ID"""
        fixed = {"field_id", "week_year"}
        series_cols = [c for c in weekly.columns if c not in fixed]

        if not series_cols:
            return weekly

        week_tuples = weekly["week_year"].map(self._parse_week_year)

        def span_key(col: str):
            """Get (min_week, max_week) span for non-zero values"""
            s = weekly[col]
            mask = s.ne(0) & s.notna()
            if not mask.any():
                return ("ALL_ZERO", None, None)
            w = week_tuples[mask]
            return ("SPAN", min(w), max(w))

        groups = {}
        for c in series_cols:
            groups.setdefault(span_key(c), []).append(c)

        to_drop = []
        for cols in groups.values():
            if len(cols) > 1:
                keep = max(cols, key=lambda x: int(str(x)))
                to_drop.extend([c for c in cols if c != keep])

        return weekly.drop(columns=to_drop)

    def fetch_harvests_weekly(
        self, field_id: int, variety_id: int, field_name: str, variety_name: str
    ) -> pd.DataFrame:
        """Fetch and aggregate harvests by week"""
        harvests = api.get_harvests(
            variety_id=variety_id,
            customer_id=self.farm_id,
            start_date=str(self.entry_date),
            end_date=str(datetime.today().date()),
        )
        harvests_df = pd.DataFrame(harvests)

        if harvests_df.empty:
            return pd.DataFrame()

        harvests_df["date"] = pd.to_datetime(harvests_df["date"])
        harvests_df["harvest_week"] = self._to_week_year(harvests_df["date"])

        if "field" in harvests_df.columns:
            harvests_df = harvests_df[harvests_df["field"] == field_name].copy()

        # Find harvest value column
        harvest_value_col = None
        for candidate in ["field_harvests", "harvest_total", "harvests", "value"]:
            if candidate in harvests_df.columns:
                harvest_value_col = candidate
                break

        if harvest_value_col is None:
            num_cols = harvests_df.select_dtypes("number").columns.tolist()
            if not num_cols:
                return pd.DataFrame()
            harvest_value_col = num_cols[0]

        weekly_h = harvests_df.groupby(["harvest_week"], as_index=False).agg(
            field_harvests=(harvest_value_col, "sum"),
        )
        weekly_h["field_id"] = field_id
        weekly_h["field"] = field_name
        weekly_h["variety_id"] = variety_id
        weekly_h["variety_name"] = variety_name

        return weekly_h

    def build_field_varieties(self) -> pd.DataFrame:
        """Build field-variety mapping from API"""
        varieties = pd.DataFrame(api.get_varieties(farm_id=self.farm_id))
        fields = api.get_fields(customer_id=self.farm_id)
        fields = pd.DataFrame(fields)

        fields = fields[["field", "field_id", "variety_id", "area_msqr"]]
        varieties = varieties[["variety_id", "variety_name"]]

        field_varieties = fields.merge(varieties, on="variety_id")
        field_varieties = field_varieties[
            field_varieties["variety_name"] != "Hass Avocado"
        ]

        logger.info(f"Loaded {len(field_varieties)} fields with varieties")
        return field_varieties

    def _build_long_for_kind(
        self,
        field_id: int,
        field_name: str,
        variety_id: int,
        variety_name: str,
        forecasts_weekly: pd.DataFrame,
        harvests: pd.DataFrame,
        kind: str,
    ) -> pd.DataFrame:
        """Build long-format table by mapping forecast series to horizons"""
        if forecasts_weekly.empty or harvests.empty:
            return pd.DataFrame()

        fixed_cols = {"field_id", "week_year"}
        series_cols = [c for c in forecasts_weekly.columns if c not in fixed_cols]

        if not series_cols:
            return pd.DataFrame()

        rows = []

        for series_id in series_cols:
            series_data = forecasts_weekly[["week_year", series_id]].copy()
            series_data = series_data[
                series_data[series_id].notna() & (series_data[series_id] != 0)
            ]

            if series_data.empty:
                continue

            series_data = series_data.sort_values("week_year")

            forecast_week = series_data.iloc[0]["week_year"]

            for idx, (_, row) in enumerate(series_data.iterrows()):
                horizon = idx + 1

                if horizon > self.max_horizon:
                    break

                target_week = row["week_year"]
                forecast_value = row[series_id]

                if pd.isna(forecast_value) or forecast_value == 0:
                    continue

                harvest_match = harvests[harvests["harvest_week"] == target_week]
                if harvest_match.empty:
                    continue

                harvest_value = harvest_match.iloc[0]["field_harvests"]
                if pd.isna(harvest_value) or harvest_value <= 0:
                    continue

                rows.append(
                    {
                        "field_id": field_id,
                        "field": field_name,
                        "variety_id": variety_id,
                        "variety_name": variety_name,
                        "forecast_week": forecast_week,
                        "target_week": target_week,
                        "horizon": horizon,
                        "forecast": forecast_value,
                        "field_harvests": harvest_value,
                        "forecast_kind": kind,
                        "series_id": str(series_id),
                    }
                )

        if not rows:
            return pd.DataFrame()

        long = pd.DataFrame(rows)

        long = long.sort_values("forecast_week")

        long["series_id_num"] = long["series_id"].astype(int)
        long = long.sort_values("series_id_num", ascending=False)
        long = long.drop_duplicates(
            subset=["field_id", "forecast_week", "horizon"], keep="first"
        )
        long = long.drop(columns=["series_id", "series_id_num"])

        long["abs_error"] = (long["forecast"] - long["field_harvests"]).abs()
        long
        return long

    def build_long_for_field(self, field_row: pd.Series) -> pd.DataFrame:
        """Build long table for a single field"""
        field_id = int(field_row["field_id"])
        field_name = str(field_row["field"])
        variety_id = int(field_row["variety_id"])
        variety_name = str(field_row["variety_name"])

        # Fetch data
        forecasts_weekly = self._fetch_forecasts_weekly(
            field_id, kind=self.forecast_kind
        )

        harvests = self.fetch_harvests_weekly(
            field_id, variety_id, field_name, variety_name
        )

        if forecasts_weekly.empty or harvests.empty:
            return pd.DataFrame()

        return self._build_long_for_kind(
            field_id,
            field_name,
            variety_id,
            variety_name,
            forecasts_weekly,
            harvests,
            kind=self.forecast_kind,
        )

    def calculate_field_performance(self, long_all: pd.DataFrame) -> pd.DataFrame:
        """Calculate field-level WAPE and bias (long format)"""
        field_metrics = long_all.groupby(
            [
                "field",
                "variety_name",
                "forecast_week",
                "horizon",
                "target_week",
                "forecast_kind",
            ],
            as_index=False,
        ).agg(
            total_forecast=("forecast", "sum"),
            total_harvest=("field_harvests", "sum"),
            abs_error_sum=("abs_error", "sum"),
        )

        field_metrics["wape"] = (
            field_metrics["abs_error_sum"] / field_metrics["total_harvest"]
        )
        field_metrics["bias"] = (
            field_metrics["total_forecast"] - field_metrics["total_harvest"]
        ) / field_metrics["total_harvest"]

        field_metrics["accuracy"] = (
            field_metrics["total_forecast"] / field_metrics["total_harvest"]
        )

        field_metrics["forecast_stability_index"] = (
            field_metrics.groupby(['field', 'variety_name', 'target_week'])['total_forecast']
            .transform(lambda x: x.std() / x.mean())
        )

        field_metrics = field_metrics[
            [
                "field",
                "variety_name",
                "forecast_week",
                "horizon",
                "target_week",
                "wape",
                "accuracy",
                "forecast_stability_index",
                "bias",
                "total_forecast",
                "total_harvest",
                "forecast_kind",
            ]
        ].rename(columns={"field": "field_name", "target_week": "horizon_week"})

        return field_metrics.sort_values(
            ["variety_name", "field_name", "forecast_week", "horizon"]
        )

    def calculate_variety_performance(self, long_all: pd.DataFrame) -> pd.DataFrame:
        """Calculate variety-level WAPE and bias (long format)"""
        variety_metrics = long_all.groupby(
            [
                "variety_name",
                "forecast_week",
                "horizon",
                "target_week",
                "forecast_kind",
            ],
            as_index=False,
        ).agg(
            total_forecast=("forecast", "sum"),
            total_harvest=("field_harvests", "sum"),
            abs_error_sum=("abs_error", "sum"),
            n_fields=("field_id", "nunique"),
        )

        variety_metrics["wape"] = (
            variety_metrics["abs_error_sum"] / variety_metrics["total_harvest"]
        )
        variety_metrics["bias"] = (
            variety_metrics["total_forecast"] - variety_metrics["total_harvest"]
        ) / variety_metrics["total_harvest"]

        variety_metrics["accuracy"] = (
            variety_metrics["total_forecast"] / variety_metrics["total_harvest"]
        )

        variety_metrics["forecast_stability_index"] = (
            variety_metrics.groupby(['variety_name', 'target_week'])['total_forecast']
            .transform(lambda x: x.std() / x.mean())
        )

        variety_metrics = variety_metrics[
            [
                "variety_name",
                "forecast_week",
                "horizon",
                "target_week",
                "wape",
                "accuracy",
                "forecast_stability_index",
                "bias",
                "total_forecast",
                "total_harvest",
                "forecast_kind",
                "n_fields",
            ]
        ].rename(columns={"target_week": "horizon_week"})

        return variety_metrics.sort_values(["variety_name", "forecast_week", "horizon"])

    def calculate_outputs(self):
        """Main method to calculate all performance metrics"""
        logger.info(f"Processing farm_id={self.farm_id}, entry_date={self.entry_date}")

        fields_df = self.build_field_varieties()

        longs = []
        for _, row in fields_df.iterrows():
            try:
                logger.info(
                    f"Processing field={row['field']} (id={row['field_id']}), variety={row['variety_name']}"
                )
                ldf = self.build_long_for_field(row)
                if not ldf.empty:
                    longs.append(ldf)
                    logger.info(f"  → Created {len(ldf)} horizon rows")
            except Exception as e:
                logger.exception(
                    f"Failed field {row.get('field_id')} {row.get('field')}: {e}"
                )

        long_all = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame()

        if long_all.empty:
            logger.warning("No comparable rows found. Writing empty outputs.")
            return pd.DataFrame(), pd.DataFrame()

        logger.info(f"Total rows collected: {len(long_all)}")
        logger.info(f"Horizons present: {sorted(long_all['horizon'].unique())}")

        long_all = long_all.sort_values("forecast_week")

        # Calculate performance metrics
        field_performance = self.calculate_field_performance(long_all)
        variety_performance = self.calculate_variety_performance(long_all)

        kr_summary = variety_performance.pivot_table(
            index="variety_name",
            columns=["horizon", "horizon_week"],
            values=["wape", "accuracy"],
            aggfunc="mean",
        )

        kr_summary.columns = [
            f"{metric}_+{h}_{wk}" for metric, h, wk in kr_summary.columns
        ]

        kr2_summary = variety_performance[
            (variety_performance["horizon_week"] >= self.start_week)
            & (variety_performance["horizon_week"] <= self.horizon_week)
        ]

        kr2_report = kr2_summary.groupby(
            ["variety_name", "horizon"],
        ).agg({"wape": "mean", "accuracy": "mean"})

        kr2_report_ = kr2_report.pivot_table(
            index="variety_name",
            columns="horizon",
            values=["wape", "accuracy"],
            aggfunc="mean",
        )

        kr2_report_.columns = [f"{metric}_+{h}" for metric, h, in kr2_report_.columns]
        field_performance = field_performance[
            field_performance["horizon_week"] == self.horizon_week
        ]
        variety_performance = variety_performance[
            variety_performance["horizon_week"] == self.horizon_week
        ]
        if variety_performance.empty:
            logger.warning(
                f"No variety_performance rows for horizon_week={self.horizon_week}. "
            )
            return field_performance, pd.DataFrame(), pd.DataFrame()
        # create summaries
        kind = variety_performance["forecast_kind"].iloc[0]
        horizon_summary = variety_performance.pivot_table(
            values=["wape", "accuracy"],
            index="variety_name",
            columns="horizon",
            aggfunc="mean",
        )
        horizon_summary.columns = [
            f"w+{h}_{kind}_{metric}" for metric, h in horizon_summary.columns
        ]

        # Save outputs
        horizon_summary= horizon_summary.reset_index()
        self._save_outputs(
            field_performance,
            variety_performance,
            kr_summary,
            kr2_report_,
            horizon_summary,
        )

        return field_performance, variety_performance, horizon_summary

    def _save_outputs(
        self,
        field_performance: pd.DataFrame,
        variety_performance: pd.DataFrame,
        kr_summary: pd.DataFrame,
        kr2_summary: pd.DataFrame,
        horizon_summary: pd.DataFrame,
    ):
        """Save performance tables to CSV"""
        os.makedirs(self.output_dir, exist_ok=True)

        farm_map = {4: "AAA", 5: "SIAN", 6: "MEO", 19: "GTF", 20: "RFF"}
        farm_name = farm_map.get(self.farm_id, "UNKNOWN")
        base = f"{farm_name}_{self.entry_date}_{self.forecast_kind}"

        field_path = os.path.join(self.output_dir, f"{base}_field_performance.csv")
        variety_path = os.path.join(self.output_dir, f"{base}_variety_performance.csv")
        variety_summary_path = os.path.join(
            self.output_dir, f"{base}_horizon_summary.csv"
        )
        kr_summary_path = os.path.join(self.output_dir, f"{base}_kr_summary.csv")
        kr2_summary_path = os.path.join(
            self.output_dir, f"{base}_KR2_variety_summary.xlsx"
        )

        #kr2_summary.to_excel(kr2_summary_path)
        #kr_summary.to_csv(kr_summary_path, index=True)
        field_performance.to_csv(field_path, index=False)
        variety_performance.to_csv(variety_path, index=False)
        horizon_summary.to_csv(variety_summary_path, index=False)

        logger.info(f"Saved field performance: {field_path}")
        logger.info(f"Saved variety performance: {variety_path}")


if __name__ == "__main__":
    for farm in [20,19,6,5,4]:
        for kind in ["raw", "final"]:
            logger.debug(f"Processing FARM - {farm}")
            horn = HorizonPerformance(
                farm_id=farm,
                horizon_week="2026-W05",
                max_horizon=4,
                forecast_kind=kind,
                output_dir="2026-W05",
            )
            field_df, variety_df, horizon_df = horn.calculate_outputs()

            print("\n" + "=" * 80)
            print("HORIZON PERFORMANCE")
            print("=" * 80)
            print(horizon_df)

            print("\n" + "=" * 80)
            print("VARIETY PERFORMANCE")
            print("=" * 80)
            print(variety_df)

            print("\n" + "=" * 80)
            print("FIELD PERFORMANCE (sample)")
            print("=" * 80)
            print(field_df.head(20))  
            print(f"Completed processing for farm {farm} with forecast kind '{kind}'")