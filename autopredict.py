"""VIN Team project 3."""

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


def celsius2farenheit(c: float) -> float:
    """Convert temperature in Farenheit to Celsius."""
    f = c * 1.8 + 32
    return f


if __name__ == "__main__":
    filename = r"data/model.sav"
    today = humanize_time(datetime.now())
    load_model = joblib.load(open(filename, "rb"))
    while True:
        temperature_str = input("Insert temperature value in Celsius: ")
        try:
            temperature = celsius2farenheit(float(temperature_str))
        except Exception:
            print("Wrong temperature value! Try again.")
            continue
        y_pred = load_model.predict(
            pd.DataFrame(
                [[temperature, today["DayOfYear"], today["TimeOfDay(s)"]]],
                columns=["Temperature", "DayOfYear", "TimeOfDay(s)"],
            )
        )
        print(f"Solar radiation is {y_pred[0]:.3f}")
