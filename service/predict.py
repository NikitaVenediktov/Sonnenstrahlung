"""VIN Team project 3."""

import argparse
from datetime import datetime

import joblib
import pandas as pd


def humanize_time(time: datetime) -> dict:
    """Add time stamps to datset."""
    res = {}
    res["MonthOfYear"] = int(time.strftime("%m"))
    res["DayOfYear"] = int(time.strftime("%j"))
    res["WeekOfYear"] = int(time.strftime("%U"))
    res["TimeOfDay(h)"] = time.hour
    res["TimeOfDay(m)"] = time.hour * 60 + time.minute
    res["TimeOfDay(s)"] = time.hour * 60 * 60 + time.minute * 60 + time.second
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        help="Temperature",
        required=True,
    )

    parser.add_argument(
        "-d",
        "--date",
        type=datetime,
        help="Date/time in datetime format",
        default=datetime.now(),
    )

    args = parser.parse_args()

    filename = r"data/model.sav"
    today = humanize_time(args.date)
    load_model = joblib.load(open(filename, "rb"))
    y_pred = load_model.predict(
        pd.DataFrame(
            [[args.temperature, today["DayOfYear"], today["TimeOfDay(s)"]]],
            columns=["Temperature", "DayOfYear", "TimeOfDay(s)"],
        )
    )
    print(f"{y_pred[0]:.3f}")
