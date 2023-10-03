import shap

import seaborn as sns
import matplotlib.pyplot as plt


def plot_detailed(rmses, target, plot_path, category):
    """
    Generate and save detailed plots based on the given category.

    This function generates and saves plots based on the provided category. It uses a specific color palette for sectors
    for consistency. The available categories are: "Revenuebucket", "Region", "Country", "Industry", "SubSector", "Year",
    "ENEConsume", and "ENEProduce". If an unsupported category is provided, it prints an error message.

    Parameters:
    - rmses (list): A list of root mean square error values for each category.
    - target (str): The target variable for the plots.
    - plot_path (str): The path where the generated plot should be saved.
    - category (str): The category for which the plot should be generated.

    Returns:
    None
    """
    sector_colors = ["#6FAA96", "#5AC2CC", "#D1E4F2", "#2E73A9", "#003f5c"]
    if category == "revenue_bucket":
        plot_revenue_bucket(rmses, target, plot_path, sector_colors)
    elif category == "region":
        plot_region(rmses, target, plot_path, sector_colors)
    elif category == "country":
        plot_country(rmses, target, plot_path, sector_colors)
    elif category == "industry":
        plot_industry(rmses, target, plot_path, sector_colors)
    elif category == "sub_sector":
        plot_sub_sector(rmses, target, plot_path, sector_colors)
    elif category == "year":
        plot_year(rmses, target, plot_path, sector_colors)
    elif category == "energy_consumed":
        plot_energy_consumed_log(rmses, target, plot_path, sector_colors)
    elif category == "energy_produced":
        plot_energy_produced_log(rmses, target, plot_path, sector_colors)
    else:
        print("Wrong category, error")
    return


def plot_revenue_bucket(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different revenue buckets.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different revenue buckets.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping revenue buckets to their respective colors.

    """
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(
        x="revenue_bucket", y=rmses.rmses, data=rmses, palette=sector_colors
    )
    plt.title(f"RMSEs box plot for {target} per revenue buckets")
    plt.xlabel("Revenue Bucket")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"revenue_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_region(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different region.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different region.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping region to their respective colors.

    """
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="region", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per regions")
    plt.xlabel("Regions")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"region_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_country(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different countries.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different countries.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping countries to their respective colors.

    """
    countries_to_keep = (
        rmses.country.unique().tolist()[:5] + rmses.country.unique().tolist()[-5:]
    )
    rmses_filtered = rmses[rmses.country.isin(countries_to_keep)]
    rmses_filtered["country"] = rmses_filtered["country"].astype(str)
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(
        x="country", y=rmses_filtered.rmses, data=rmses_filtered, palette=sector_colors
    )
    plt.title(f"RMSEs box plot for {target} per countries (5 bests and 5 worses)")
    plt.xlabel("Countries")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"country_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_industry(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different industries.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different industries.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping industries to their respective colors.

    """
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="industry", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per industry")
    plt.xlabel("Industries")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"industry_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_sub_sector(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different sectors.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different sectors.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping sectors to their respective colors.

    """
    sectors_to_keep = (
        rmses.sub_sector.unique().tolist()[:5] + rmses.sub_sector.unique().tolist()[-5:]
    )
    rmses_filtered = rmses[rmses.sub_sector.isin(sectors_to_keep)]
    rmses_filtered["sub_sector"] = rmses_filtered["sub_sector"].astype(str)

    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(
        x="sub_sector",
        y=rmses_filtered.rmses,
        data=rmses_filtered,
        palette=sector_colors,
    )
    plt.title(f"RMSEs box plot for {target} per sub-sectors (5 bests and 5 worses)")
    plt.xlabel("Sectors")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"sector_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_year(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different years.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different years.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping years to their respective colors.

    """
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(x="year", y=rmses.rmses, data=rmses, palette=sector_colors)
    plt.title(f"RMSEs box plot for {target} per years")
    plt.xlabel("years")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"years_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_energy_consumed_log(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different energy consumed.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different energy consumed.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping energy consumed to their respective colors.

    """
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(
        x="energy_consumed", y=rmses.rmses, data=rmses, palette=sector_colors
    )
    plt.title(f"RMSEs box plot for {target} depending on energy_consumed availability")
    plt.xlabel("Energy Consumed")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"consume_box_plot_{target}.png", bbox_inches="tight")
    return


def plot_energy_produced_log(rmses, target, plot_path, sector_colors):
    """
    Generate a box plot of Root Mean Square Errors (RMSEs) for a given target variable
    across different energy produced.

    Parameters:
        rmses (DataFrame): A DataFrame containing RMSE values for different energy produced.
        target (str): The target variable for which RMSEs are being plotted.
        plot_path (str): The path where the generated plot image will be saved.
        sector_colors (dict): A dictionary mapping energy produced to their respective colors.

    """
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(
        x="energy_produced", y=rmses.rmses, data=rmses, palette=sector_colors
    )
    plt.title(f"RMSEs box plot for {target} depending on energy_produced availability")
    plt.xlabel("Energy Produced")
    plt.ylabel("RMSEs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(plot_path + f"produce_box_plot_{target}.png", bbox_inches="tight")
    return


def plot(model, X, y_test, y_pred, plot_path, target):
    """
    This function generates three plots for evaluating the machine learning model's performance:

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

    Returns:
    Plots saved in plot_path
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
        plt.figure()
        shap.plots.beeswarm(explanation, show=False, color_bar=False)
        plt.colorbar()
        plt.savefig(plot_path + f"shap_{target}.png", bbox_inches="tight")
        plt.close()
        return

    def plot_y_test_y_pred(y_test, y_pred):
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.savefig(plot_path + f"y_test_y_pred_{target}.png", bbox_inches="tight")
        plt.close()
        return

    def plot_residuals(y_test, y_pred):
        plt.figure()
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color="r", linestyle="-")
        plt.title("Residual Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.savefig(plot_path + f"residus_{target}.png", bbox_inches="tight")
        plt.close()
        return

    plot_shap_values(model, X)
    plot_y_test_y_pred(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    return
