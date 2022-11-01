"""VIN Team project 3."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pytz import timezone
import pytz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.dates as md
from typing import List


def humanize_time(
    data: pd.DataFrame, tzone: str, timeformat: str = "UNIXTime", inline: bool = False
) -> pd.DataFrame:
    """Add time stamps to datset."""
    dataset = data if inline else data.copy()
    dataset["date_for_plot"] = pd.to_datetime(dataset[timeformat], unit="s")
    dataset.index = dataset.index.tz_localize(pytz.utc).tz_convert(timezone(tzone))
    dataset["MonthOfYear"] = dataset.index.strftime("%m").astype(int)
    dataset["DayOfYear"] = dataset.index.strftime("%j").astype(int)
    dataset["WeekOfYear"] = dataset.index.strftime("%U").astype(int)
    dataset["TimeOfDay(h)"] = dataset.index.hour
    dataset["TimeOfDay(m)"] = dataset.index.hour * 60 + dataset.index.minute
    dataset["TimeOfDay(s)"] = (
        dataset.index.hour * 60 * 60 + dataset.index.minute * 60 + dataset.index.second
    )
    dataset["TimeSunRise"] = pd.to_datetime(dataset["TimeSunRise"], format="%H:%M:%S")
    dataset["TimeSunSet"] = pd.to_datetime(dataset["TimeSunSet"], format="%H:%M:%S")
    dataset["DayLength(s)"] = (
        dataset["TimeSunSet"].dt.hour * 60 * 60
        + dataset["TimeSunSet"].dt.minute * 60
        + dataset["TimeSunSet"].dt.second
        - dataset["TimeSunRise"].dt.hour * 60 * 60
        - dataset["TimeSunRise"].dt.minute * 60
        - dataset["TimeSunRise"].dt.second
    )
    return dataset


def plot_diags(
    dataset: pd.DataFrame,
    params: List[str],
    periods: List[str],
    palette: str = "YlOrRd_r",
) -> plt.Figure:
    """Plots diagrams."""
    assert set(params) <= set(dataset.columns), "Params must be in dataset columns"
    assert set(periods) <= set(dataset.columns), "Periods must be in dataset columns"

    grouped = {p: dataset.groupby(p).mean().reset_index() for p in periods}
    f, axes = plt.subplots(
        len(params), len(periods), sharex="col", sharey="row", figsize=(14, 12)
    )
    for i, param in enumerate(params):
        for j, period in enumerate(periods):
            axes[i, j].set_title(f"Mean {param.lower()} by {period}")
            pal = sns.color_palette(palette, len(grouped[period]))
            rank = grouped[period][param].argsort().argsort()
            sns.barplot(
                x=period,
                y=param,
                data=grouped[period],
                palette=np.array(pal[::-1])[rank],
                ax=axes[i, j],
            )
            axes[i, j].set_xlabel("")
    plt.show()
    return f


def main():
    """This function makes every code better."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--file_name",
        type=str,
        help="Path to a dataset",
        default=r"data/SolarPrediction.csv",
    )

    parser.add_argument(
        "-z",
        "--time_zone",
        type=str,
        help="Time zone",
        default=r"Pacific/Honolulu",
    )

    parser.add_argument(
        "-t",
        "--time_format",
        type=str,
        help="Time format",
        default="UNIXTime",
    )

    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot diagrams",
    )

    args = parser.parse_args()

    # Preparing data
    dataset = pd.read_csv(args.file_name)
    dataset = dataset.sort_values([args.time_format], ascending=[True])
    dataset.head()
    dataset.index = pd.to_datetime(dataset[args.time_format], unit="s")
    dataset = humanize_time(
        dataset,
        tzone=args.time_zone,
    )
    dataset.drop(["Data", "Time", "TimeSunRise", "TimeSunSet"], inplace=True, axis=1)

    # Plot diagrams
    if args.plot:
        plot_diags(
            dataset,
            params=["Radiation", "Pressure", "Humidity", "Pressure"],
            periods=["TimeOfDay(h)", "MonthOfYear"],
        )
        corrmat = dataset.drop(
            [
                "TimeOfDay(h)",
                "TimeOfDay(m)",
                "TimeOfDay(s)",
                "UNIXTime",
                "MonthOfYear",
                "WeekOfYear",
            ],
            inplace=False,
            axis=1,
        )
        corrmat = corrmat.corr()
        f, ax = plt.subplots(figsize=(7, 7))
        sns.heatmap(corrmat, vmin=-0.8, vmax=0.8, square=True, cmap="coolwarm")
        plt.show()

    x = dataset[
        [
            "Temperature",
            "Pressure",
            "Humidity",
            "WindDirection(Degrees)",
            "Speed",
            "DayOfYear",
            "TimeOfDay(s)",
        ]
    ]
    y = dataset["Radiation"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )

    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(x_train, y_train)
    feature_importances = regressor.feature_importances_

    x_train_opt = x_train.copy()
    removed_columns = pd.DataFrame()
    models = []
    r2s_opt = []

    for _ in range(0, 5):
        least_important = np.argmin(feature_importances)
        removed_columns = removed_columns.append(
            x_train_opt.pop(x_train_opt.columns[least_important])
        )
        regressor.fit(x_train_opt, y_train)
        feature_importances = regressor.feature_importances_
        accuracies = cross_val_score(
            estimator=regressor, X=x_train_opt, y=y_train, cv=5, scoring="r2"
        )
        r2s_opt = np.append(r2s_opt, accuracies.mean())
        models = np.append(models, ", ".join(list(x_train_opt)))

    feature_selection = pd.DataFrame({"Features": models, "r2 Score": r2s_opt})
    feature_selection.head()

    x_train_best = x_train[["Temperature", "DayOfYear", "TimeOfDay(s)"]]
    x_test_best = x_test[["Temperature", "DayOfYear", "TimeOfDay(s)"]]
    regressor.fit(x_train_best, y_train)

    accuracies = cross_val_score(
        estimator=regressor, X=x_train_best, y=y_train, cv=10, scoring="r2"
    )
    accuracy = accuracies.mean()
    print(f"r2 = {accuracy}")

    y_pred = regressor.predict(x_test_best)
    r_squared = r2_score(y_test, y_pred)
    print(f"r2 = {r_squared}")

    dataset["y_pred"] = regressor.predict(
        dataset[["Temperature", "DayOfYear", "TimeOfDay(s)"]]
    )

    if args.plot:
        fig = plt.figure(figsize=(30, 6))
        ax = fig.add_subplot()
        xfmt = md.DateFormatter("%m-%d %H:%M:%S")
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=25)
        plt.xlabel("l")
        ax.xaxis_date()
        plt.plot(
            dataset.date_for_plot[:300],
            dataset.Radiation[:300],
            label="Наблюдаемая радиация",
        )
        plt.plot(
            dataset.date_for_plot[:300],
            dataset.y_pred[:300],
            label="Предсказанная радиация",
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
