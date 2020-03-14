
# CoViD-19 Data Analysis

Some data analysis in python around the covid-19 data (including survival analysis with Kaplan-Meier).

Data Source: [CSSEGISandData/COVID-19](https://github.com/CSSEGISandData/COVID-19), the data behind the dashboard: [Coronavirus COVID-19 Global Cases by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU)](https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html#/bda7594740fd40299423467b48e9ecf6).

The first notebook just visualizes the numbers on a daily basis for a few regions: [covid-19-data-analysis.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/covid-19-data-analysis.ipynb?flush_cache=true)

The second notebook does some survival analysis via Kaplan-Meier: [covid-19-survival.ipynb](https://nbviewer.jupyter.org/github/cs224/covid-19-data-analysis/blob/master/covid-19-survival.ipynb?flush_cache=true)

Read more about the background in the associated blog post here: [CoViD-19 Data Analysis](https://weisser-zwerg.dev/posts/covid-19-data-analysis/).

## Note / Caveat

It seems that the `?flush_cache=true` flag for [jupyter nbviewer](https://nbviewer.jupyter.org/) does not work any longer. Therefore by clicking on the above links you may get **outdated results**.
You can easily spot this if the `last updated:` at the very top of the notebook is older than the last commit (just compare this value to the values you see when clicking on the *github* links below).

To be sure you get the latest versions you can either look at the notebooks on *github* (which are not so pretty):
* [covid-19-data-analysis.ipynb](https://github.com/cs224/covid-19-data-analysis/blob/master/covid-19-data-analysis.ipynb)
* [covid-19-survival.ipynb](https://github.com/cs224/covid-19-data-analysis/blob/master/covid-19-survival.ipynb)

Or you checkout the repository and run the notebooks locally on your machine.
