# CGEE - Corporate GHG Emissions Estimations

- [About CGEE 💡](#about)

  - [Context](#context)
  - [The Pladifes project](#pladifes)

- [Quickstart 🚀](#quickstart)
  - [Installation ](#installation)
    - [Getting the code](#get)
    - [Dependencies](#dependencies)
  - [Utilisation](#utilization)
    - [Repository organization](#orga)
    - [Pipeline](#pipeline)
    - [Access to base models performances](#perfs)
    - [Models customization](#custom)
- [References 📝](#refs)
- [Contributing 🤝](#contributing)
- [Contact ✉️](#contact)

# <a id="about"></a> About CGEE

## <a id="context"></a> Context

This project aims to estimate <b>corporate greenhouse gas (GHG) emissions</b> using an open methodology based on state-of-the-art literature.

Corporate GHG emissions are a critical issue for understanding and assessing real company transition strategies effectiveness. It is a key indicator to derive <b>transition risk</b> and <b>impact on climate change</b>. As regulation evolves, it is becoming more and more mandatory for companies to disclose their scope 1, 2 and 3 emissions, but the coverage is far from being close to exhaustivity.

To deal with unreported companies, researchers and practitionners rely on estimated data, ususally from well established data providers. However, these estimates :

- Often rely on a <b>non-transparent methodology</b>,
- May not always be <b>accurate</b> (usually based on sectorial means),
- Can be <b>expensive</b> (>10k $).

**CGEE** (standing for corporate GHG emissions estimations) address these challenges, by leveraging the latest research and best practices to develop a open data science methodology that allows estimating GHG emissions (Scope 1, Scope 2, Scope 3 and Scope 123) accurately using machine learning.

We plan to release an online calculator on our [dedicated website](https://pladifes.institutlouisbachelier.org/) in 2023 and already give free access to researchers to our <b>complete datasets</b>.

## <a id="pladifes"></a> Pladifes

Pladifes is a research program aiming at easing the research in green and sustainable finance, as well as traditional one. They are the main authors of <b>CGEE</b>. Learn more about Pladifes [here](https://www.institutlouisbachelier.org/en/pladifes-a-large-financial-and-extra-financial-database-project-2/).

Databases produced in the context of this project are available [here](https://pladifes.institutlouisbachelier.org/data/#ghg-estimations). Other Paldifes databases can be access [here (only ESG)](https://pladifes.institutlouisbachelier.org/data/) and [here (financial and soon ESG)](https://www.eurofidai.org/).

# <a id="quickstart"></a> Quickstart 🚀

## <a id="installation"></a> Installation

### <a id="get"></a> Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    https://github.com/Pladifes/corporate-ghg-emissions-estimations.git

### <a id="dependencies"></a> Dependencies

You'll need a working Python environment to run the code.
The required dependencies are specified in the file `Pipfile`.
We use `pipenv` virtual environments to manage the project dependencies in
isolation.

Thus, you can install our dependencies without causing conflicts with your
setup (for a Python >= 3.9).
Run the following command in the repository folder (where `Pipfile` and `Pipfile.lock`
is located) to create a separate environment.
Make sure pipenv is installed on your system. You can install it using pip by running:

    pip install pipenv

to install all required dependencies in it:

    pipenv install

Once the installation is complete, activate the virtual environment by running:

    pipenv shell

This will activate the virtual environment and you can start working with your installed dependencies.

## <a id="utilization"></a> Utilization

### <a id="orga"></a> Repository organization

<mark>Benchmark</mark> explains how we create our test dataset to evaluate our models. We invite anyone working on this subject to compare your performances with our pre selected companies and metrics.

<mark>Basic_Model_training.ipynb</mark> In this notebook, the code is executed with predefined parameters as specified during the study.

<mark>Advanced_Model_training.ipynb</mark> This notebook provides a comprehensive overview of the code execution and allows you to test the pipeline with different parameters.

Both notebooks are divided into two main sections:
- Training with Unrestricted Features: This section utilizes a wide range of open and proprietary data sources to build the models on.

- Training with Restricted Features (Forbes Data): This section uses the same data sources but with fewer features.

<mark>Unrestricted features</mark> This section utilizes a wide range of open and proprietary data sources to build the models:

- International Energy Agency (<b>IEA</b>) [GHG emission data from energy data](https://www.iea.org/data-and-statistics/data-tools/greenhouse-gas-emissions-from-energy-data-explorer), including free fuel mix carbon intensity data.
- [<b>Ember</b> Yearly Electricity data](https://ember-climate.org/data-catalogue/yearly-electricity-data/), emissions associated with electricity géneration.
- <b>WorldBank</b> data, including information about [countries income levels](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups) and [CO2 laws](https://carbonpricingdashboard.worldbank.org/map_data).
- <b>CDP</b> and <b>Refinitiv</b> data, proprietary data used for corporate financial, energy and emission data.


<mark>Restricted features</mark> This section uses the same data sources but with fewer features. Using less features allows  for applying saved models on proposed Forbes 2000 data, to see how applications can be made. 




<!-- ### <a id="pipeline"></a> Pipeline

Once the working environment has been successfully installed, the pipeline can be initiated in two ways. The first method involves using the file ****main**.py** in a global manner by setting the variable `Open_data` to <b>True</b> to run the Open data pipeline and <b>False</b> to run the propriatary pipeline . Alternatively, for a more detailed approach, you can use each pipeline Jupyter Notebook where you can perform various experiments and monitor them using <b>mlflow</b>. This allows you to easily track and compare different experiments, which can help you optimize the pipeline's performance according to your datasets and objectifs. -->

## <a id="perfs"></a> Access to base models performances

The methodology being used involves training four distinct models for each target. For each one, the model with the best RMSE score on a pre-defined out-of-sample set of companies is selected. This top-performing model is then saved as a pickle file (.pkl) in a folder called "models/".

## <a id="custom"></a> Models customization

Please reach out to us if you want to discuss how to modify our models according to your project.

# <a id="refs"></a> References

Our approch is highly inspired from the following publications and discussion with some of the main authors:

- [ `Quyen Nguyen, Ivan Diaz-Rainey, and Duminda Kuruppuarachchi. Predicting corporate carbon footprints for climate finance risk analyses: a machine learning approach. Energy Economics,2021.` ](https://econpapers.repec.org/article/eeeeneeco/v_3a95_3ay_3a2021_3ai_3ac_3as0140988321000347.htm)
- [`Quyen Nguyen, Ivan Diaz-Rainey, Adam Kitto, Ben McNeil, Nicholas A. Pittman and Renzhu Zhang. Scope 3 Emissions: Data Quality and Machine Learning Prediction Accuracy. USAEE Working Paper, 2022.`](https://deliverypdf.ssrn.com/delivery.php?ID=125017092064004021102004023083123112041074054049036036007019126101098114006075110124124027025103026004058017102077008006116080104038028011067093067017086065066126037076043124084026083001082009029083114003069019077119126091127083114019108065065122091&EXT=pdf&INDEX=TRUE)
- [`Jeremi Assael, Thibaut Heurtebize, Laurent Carlier and François Soupé. Greenhouse gases emissions: estimating corporate non-reported emissions using interpretable machine learning. Artificial Intelligence for the Sustainable Economics and Business, 2023`](https://econpapers.repec.org/paper/arxpapers/2212.10844.htm)

# <a id="contributing"></a> Contributing 🤝

We are hoping that the open-source community will help us edit the code and improve it!

You are welcome to open issues, even suggest solutions and better still contribute the fix/improvement! We can guide you if you're not sure where to start but want to help us out 🥇

In order to contribute a change to our code base, you can submit a pull request (PR) via GitLab and someone from our team will go over it. Yet, we usually prefer that you reach out to us before doing so at [contact](mailto:pladifes@institutlouisbachelier.org).

# <a id="contact"></a> Contact 📝

Maintainers are [Mohamed FAHMAOUI](https://www.linkedin.com/in/mohamed-fahmaoui-b30587176/) and [Thibaud BARREAU](https://www.linkedin.com/in/thibaud-barreau/). CGEE is developed in the context of the [Pladifes](https://pladifes.institutlouisbachelier.org/) project. If you use Pladifes repositories, models or datasets for academic research, you must mention the following sentence : "Work carried out using resources provided by the Equipex Pladifes (ANR-21-ESRE-0036), hosted by the Institut Louis Bachelier".
