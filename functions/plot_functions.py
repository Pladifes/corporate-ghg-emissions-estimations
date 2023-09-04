import shap

import seaborn as sns
import matplotlib.pyplot as plt


def plot_detailed(rmses, target, plot_path, category):
    sector_colors = ["#6FAA96", "#5AC2CC", "#D1E4F2", "#2E73A9", "#003f5c"]
    if category == "Revenuebucket":
        plot_Revenuebucket(rmses, target, plot_path, sector_colors)
    elif category == "Region":
        plot_Region(rmses, target, plot_path, sector_colors)
    elif category == "Country":
        plot_Country(rmses, target, plot_path, sector_colors)
    elif category == "Industry":
        plot_Industry(rmses, target, plot_path, sector_colors)
    elif category == "SubSector":
        plot_SubSector(rmses, target, plot_path, sector_colors)
    elif category == "Year":
        plot_Year(rmses, target, plot_path, sector_colors)
    elif category == "ENEConsume":
        plot_ENEConsume_log(rmses, target, plot_path, sector_colors)
    elif category == "ENEProduce":
        plot_ENEProduce_log(rmses, target, plot_path, sector_colors)
    else:
        print("Wrong category, error")


def plot_Revenuebucket(rmses, target, plot_path, sector_colors):
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="Revenuebucket", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per Revenue buckets")
    plt.xlabel("Revenue Bucket")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"revenue_box_plot_{target}.png", bbox_inches="tight")


def plot_Region(rmses, target, plot_path, sector_colors):
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="Region", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per regions")
    plt.xlabel("Regions")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"region_box_plot_{target}.png", bbox_inches="tight")


def plot_Country(rmses, target, plot_path, sector_colors):
    countries_to_keep = rmses.Country.unique().tolist()[:5] + rmses.Country.unique().tolist()[-5:]
    rmses_filtered = rmses[rmses.Country.isin(countries_to_keep)]
    rmses_filtered["Country"] = rmses_filtered["Country"].astype(str)
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="Country", y=rmses_filtered.rmses, data=rmses_filtered, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per countries (5 bests and 5 worses)")
    plt.xlabel("Countries")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"country_box_plot_{target}.png", bbox_inches="tight")


def plot_Industry(rmses, target, plot_path, sector_colors):
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="Industry", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per Industry")
    plt.xlabel("Industries")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"industry_box_plot_{target}.png", bbox_inches="tight")


def plot_SubSector(rmses, target, plot_path, sector_colors):
    sectors_to_keep = rmses.SubSector.unique().tolist()[:5] + rmses.SubSector.unique().tolist()[-5:]
    rmses_filtered = rmses[rmses.SubSector.isin(sectors_to_keep)]
    rmses_filtered["SubSector"] = rmses_filtered["SubSector"].astype(str)

    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="SubSector", y=rmses_filtered.rmses, data=rmses_filtered, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per sub-sectors (5 bests and 5 worses)")
    plt.xlabel("Sectors")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"sector_box_plot_{target}.png", bbox_inches="tight")


def plot_Year(rmses, target, plot_path, sector_colors):
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="Year", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per years")
    plt.xlabel("Years")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"years_box_plot_{target}.png", bbox_inches="tight")


def plot_ENEConsume_log(rmses, target, plot_path, sector_colors):
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="ENEConsume", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} depending on ENEConsume availability")
    plt.xlabel("ENE Consume")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"consume_box_plot_{target}.png", bbox_inches="tight")


def plot_ENEProduce_log(rmses, target, plot_path, sector_colors):
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="ENEProduce", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} depending on ENEProduce availability")
    plt.xlabel("ENE Produce")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"produce_box_plot_{target}.png", bbox_inches="tight")


def plot(model, X, y_test, y_pred, plot_path, target):
    """
    This function generates three plots for evaluating the machine learning model's performance :

    A SHAP values plot showing the impact of each feature on the model's predictions.
    A scatter plot comparing the actual values against the predicted values.
    A residual plot showing the distribution of the difference between actual and predicted values.
    Parameters:

    model: The trained machine learning model to be evaluated.
    X: The feature matrix for the test dataset.
    y_test: The ground truth labels for the test dataset.
    y_pred: The predicted labels for the test dataset.
    plot_path: The directory where the plots will be saved.
    target: The name of the target variable.
    model_name: The name of the machine learning model.
    Returns:
    plots saved in plot_path
    """

    def plot_shap_values(model, X):
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X)
        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X,
            feature_names=list(X.columns),
        )
        shap.plots.beeswarm(explanation, show=False, color_bar=False)
        plt.colorbar()
        plt.savefig(plot_path + f"shap_{target}.png", bbox_inches="tight")
        plt.close()

    def plot_y_test_y_pred(y_test, y_pred):
        plt.scatter(y_test, y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.savefig(plot_path + f"y_test_y_pred_{target}.png", bbox_inches="tight")
        plt.close()

    def plot_residuals(y_test, y_pred):
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color="r", linestyle="-")
        plt.title("Residual Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.savefig(plot_path + f"residus_{target}.png", bbox_inches="tight")
        plt.close()

    plot_shap_values(model, X)
    plot_y_test_y_pred(y_test, y_pred)
    plot_residuals(y_test, y_pred)