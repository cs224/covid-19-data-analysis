
# CoViD-19 Data Analysis

Some data analysis in python around the covid-19 data (including survival analysis with Kaplan-Meier).

Data Source: [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19), the data behind the dashboard: [Coronavirus COVID-19 Global Cases by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU)](https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6).

The first notebook just visualizes the numbers on a daily basis for a few regions: [covid-19-data-analysis.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/covid-19-data-analysis.ipynb?flush_cache=true)

The second notebook does some survival analysis via Kaplan-Meier: [covid-19-survival.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/covid-19-survival.ipynb?flush_cache=true)

The idea for the third notebook is from [Markus Noga](https://www.linkedin.com/in/mlnoga/)'s notebooks: [covid19-analysis](https://mlnoga.github.io/covid19-analysis/). I am fitting a sigmoid and an exponential function to the data (via [PyMC3](https://docs.pymc.io/)) for Austria and Germany and then perform a model comparison to see which of the two models works better on the data. In both cases the sigmoid model is more likely and today (2020-04-01) the model says that the inflexion point was on 2020-03-26: [covid-19-data-analysis-forecasting.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/covid-19-data-analysis-forecasting.ipynb?flush_cache=true). If the model is right we should see a max in Austria of 15'000 cases and in Germany a max of 85'000 cases.

Read more about the background in the associated blog post here: [CoViD-19 Data Analysis](https://weisser-zwerg.dev/posts/covid-19-data-analysis/).

## Note / Caveat

It seems that the `?flush_cache=true` flag for [jupyter nbviewer](https://nbviewer.jupyter.org/) does not work any longer. Therefore by clicking on the above links you may get **outdated results**.
You can easily spot this if the `last updated:` at the very top of the notebook is older than the last commit (just compare this value to the values you see when clicking on the *github* links below).

To be sure you get the latest versions you can either look at the notebooks on *github* (which are not so pretty):
* [covid-19-data-analysis.ipynb](https://github.com/cs224/covid-19-data-analysis/blob/master/covid-19-data-analysis.ipynb)
* [covid-19-survival.ipynb](https://github.com/cs224/covid-19-data-analysis/blob/master/covid-19-survival.ipynb)

Or you checkout the repository and run the notebooks locally on your machine.

# Related

## Relative risk based on age and sex derived from data from Germany (2020-03-31)

* [covid19-germany-death-rate-by-sex.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/2020-03-31-covid19-germany-death-rate-by-sex.ipynb?flush_cache=true)

## Analysis for data in Bavaria

* [covid-19-data-analysis-bavaria.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/covid-19-data-analysis-bavaria.ipynb?flush_cache=true)
