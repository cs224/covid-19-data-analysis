
import numpy as np, scipy, scipy.stats as stats, scipy.special, scipy.misc, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, xarray as xr
import matplotlib as mpl
import lifelines

import pymc3 as pm

import theano as thno
import theano.tensor as T

import datetime, time, math
from dateutil import relativedelta

from collections import OrderedDict
import sys, os

import json
import urllib.request
import yaml
import copy
import gpflow, GPy
import simdkalman
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import re

# fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
time_series_19_covid_confirmed = pd.read_csv(fname)
# deprecated
# fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
# fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/cd66d80f60dc1a8ecb6e0f362cc206c5c43b7144/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'
fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
time_series_19_covid_recovered = pd.read_csv(fname)
# fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
fname = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
time_series_19_covid_death = pd.read_csv(fname)

yesterday = datetime.date.today() - datetime.timedelta(days=1)
augment_time_series_from_daily_snapshots_date_range = pd.date_range(start='2020-03-14', end=yesterday)
augment_time_series_from_daily_snapshots_fname_pattern = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'
# disable merge from daily files, because it seems that the data update pipeline is fixed again.
augment_time_series_from_daily_snapshots_date_range = []

def fill_recovered_ds(ldf_confirmed_base, ldf_recovered_base):
    l1 = len(ldf_confirmed_base.columns)
    l2 = len(ldf_recovered_base.columns)
    fill_up_columns = list(ldf_confirmed_base.columns)[l2:]
    for c in fill_up_columns:
        ldf_recovered_base[c] = -1

    return ldf_recovered_base


def augment_time_series_from_daily_snapshots(date_list):
    d = date_list[0]
    date_column_year = d.date().strftime('%y')
    date_column = '{}/{}/{}'.format(d.date().month, d.date().day, date_column_year)
    columns_to_drop = list(time_series_19_covid_confirmed.loc[:,date_column:].columns)

    ldf_confirmed_base = time_series_19_covid_confirmed.copy()
    ldf_confirmed_base.drop(columns=columns_to_drop, inplace=True)
    ldf_recovered_base = time_series_19_covid_recovered.copy()
    ldf_recovered_base.drop(columns=columns_to_drop, inplace=True)
    ldf_death_base = time_series_19_covid_death.copy()
    ldf_death_base.drop(columns=columns_to_drop, inplace=True)

    ldf_confirmed_base.fillna(-1, inplace=True)
    ldf_recovered_base.fillna(-1, inplace=True)
    ldf_death_base.fillna(-1, inplace=True)

    for d in date_list:
        # print(d)
        ldf = pd.read_csv(augment_time_series_from_daily_snapshots_fname_pattern.format(d.date().strftime('%m-%d-%Y')))
        # '3/14/20'
        date_column_year = d.date().strftime('%y')
        date_column = '{}/{}/{}'.format(d.date().month, d.date().day, date_column_year)

        ldf_confirmed = ldf[['Province/State', 'Country/Region', 'Confirmed']].copy()
        ldf_confirmed = ldf_confirmed.rename(columns={'Confirmed': date_column})
        ldf_confirmed.fillna(-1, inplace=True)
        ldf_confirmed_base = pd.merge(ldf_confirmed_base, ldf_confirmed, how='left', on=['Province/State', 'Country/Region'])
        ldf_confirmed_base.fillna(-1, inplace=True)
        ldf_confirmed_base[date_column] = ldf_confirmed_base[date_column].astype(np.int)

        ldf_recovered = ldf[['Province/State', 'Country/Region', 'Recovered']].copy()
        ldf_recovered = ldf_recovered.rename(columns={'Recovered': date_column})
        ldf_recovered.fillna(-1, inplace=True)
        ldf_recovered_base = pd.merge(ldf_recovered_base, ldf_recovered, how='left', on=['Province/State', 'Country/Region'])
        ldf_recovered_base.fillna(-1, inplace=True)
        ldf_recovered_base[date_column] = ldf_recovered_base[date_column].astype(np.int)

        ldf_death = ldf[['Province/State', 'Country/Region', 'Deaths']].copy()
        ldf_death = ldf_death.rename(columns={'Deaths': date_column})
        ldf_death.fillna(-1, inplace=True)
        ldf_death_base = pd.merge(ldf_death_base, ldf_death, how='left', on=['Province/State', 'Country/Region'])
        ldf_death_base.fillna(-1, inplace=True)
        ldf_death_base[date_column] = ldf_death_base[date_column].astype(np.int)

    ldf_recovered_base = fill_recovered_ds(ldf_confirmed_base, ldf_recovered_base)

    return ldf_confirmed_base, ldf_recovered_base, ldf_death_base

columns = time_series_19_covid_confirmed.columns[4:]
dcolumns = [pd.to_datetime(dt) for dt in columns]
columns[:3],dcolumns[-3:]

p = os.path.realpath('.')
m = re.search('(^.*covid-19-data-analysis).*', p)
p = m.group(1)

override_xlsx_name = p + '/covid-manual-excel.xlsx'
override_xlsx = pd.ExcelFile(override_xlsx_name)

def get_cases_by_region_override(region='Germany'):
    if region not in override_xlsx.sheet_names:
        return None
    override_df = override_xlsx.parse(region, index_col=0)
    override_df.index = pd.Series(override_df.index).dt.date
    return override_df

def get_cases_by_selector(selector, region='Germany'):
    if augment_time_series_from_daily_snapshots_date_range is not None and len(augment_time_series_from_daily_snapshots_date_range) > 0:
        ldf_confirmed, ldf_recovered, ldf_death = augment_time_series_from_daily_snapshots(augment_time_series_from_daily_snapshots_date_range)
    else:
        ldf_confirmed, ldf_recovered, ldf_death = time_series_19_covid_confirmed.copy(), time_series_19_covid_recovered.copy(), time_series_19_covid_death.copy()

    # ldf_recovered = fill_recovered_ds(ldf_confirmed, ldf_recovered)

    # if region == 'US':
    #     selector = pd.Series([True])
    #     ldf_confirmed, ldf_recovered, ldf_death = get_us_data_for_time_series(time_series_19_covid_confirmed), get_us_data_for_time_series(time_series_19_covid_recovered), get_us_data_for_time_series(time_series_19_covid_death)


    # return ldf_confirmed, ldf_recovered, ldf_death, columns, selector
    ldf_recovered = pd.merge(ldf_confirmed[['Country/Region', 'Province/State']], ldf_recovered, how='left', on=['Country/Region', 'Province/State']).fillna(0.0)

    ldf_confirmed = ldf_confirmed[columns][selector]

    ldf_recovered = (ldf_recovered[columns]).astype(np.int)
    ldf_recovered = ldf_recovered[selector]

    ldf_death     = ldf_death[columns][selector]

    if (len(ldf_confirmed) > 1):
        ldf_confirmed = pd.DataFrame(ldf_confirmed.sum(axis=0), columns=['sum']).T
        ldf_recovered = pd.DataFrame(ldf_recovered.sum(axis=0), columns=['sum']).T
        ldf_death = pd.DataFrame(ldf_death.sum(axis=0), columns=['sum']).T

    # return ldf_confirmed, ldf_recovered, ldf_death

    ldf = pd.DataFrame(index=dcolumns)
    ldf['confirmed'] = ldf_confirmed.T.values
    ldf['recovered'] = ldf_recovered.T.values
    # ldf['recovered'] = 0
    ldf['death'] = ldf_death.T.values

    override_df = get_cases_by_region_override(region=region)
    if override_df is not None:
        override_df.index = pd.to_datetime(pd.Series(override_df.index) + datetime.timedelta(days=-1))
        for idx, row in override_df.iterrows():
            ldf.loc[idx] = row

    lv = ldf['confirmed'].iloc[1:].values - ldf['confirmed'].iloc[:-1].values
    ldf['new_confirmed'] = np.concatenate([np.array([0]), lv])

    lv = ldf['recovered'].iloc[1:].values - ldf['recovered'].iloc[:-1].values
    ldf['new_recovered'] = np.concatenate([np.array([0]), lv])

    lv = ldf['death'].iloc[1:].values - ldf['death'].iloc[:-1].values
    ldf['new_death'] = np.concatenate([np.array([0]), lv])

    return ldf.astype(np.int)

def get_cases_by_region(region='Germany'):

    # if region == 'Germany':
    #     return get_rki_df()

    r = region
    if isinstance(region, str):
        region = [region]

    return get_cases_by_selector(time_series_19_covid_confirmed['Country/Region'].isin(region), region=r)


class CasesByRegion():
    def __init__(self, region, df=None):
        self.region = region
        if df is None:
            self.df = get_cases_by_region(region=region)
        else:
            self.df = df

        self.df.fillna(0.0, inplace=True)

    def tail(self):
        return self.df.tail()

    def plot_daily_stats(self, ax=None, days=20):
        if ax is None:
            fig = plt.figure(figsize=(32, 8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1, 1, 1)
        last_day = self.df.index[-1]
        # self.df[['new_confirmed', 'recovered', 'death']].loc[last_day + datetime.timedelta(days=-20):].plot.bar(ax=ax)
        ax = self.df[['new_confirmed', 'new_recovered', 'new_death']].loc[last_day + datetime.timedelta(days=-days):].plot.bar(ax=ax)
        plt.tick_params(labelright=True)  # labeltop=True,
        return ax

    def plot_daily_stacked(self, days=20):
        fig = plt.figure(figsize=(32, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.subplot(1, 1, 1)
        last_day = self.df.index[-1]
        ldf_ = self.df
        ldf = ldf_[['confirmed', 'new_confirmed']].copy()
        ldf['confirmed'] = ldf['confirmed'] - ldf['new_confirmed']
        ldf[['confirmed', 'new_confirmed']].loc[last_day + datetime.timedelta(days=-days):].plot.bar(ax=ax, stacked=True)

        totals = []
        # find the values and append to list
        for i in ax.patches:
            totals.append(i.get_height())

        # set individual bar lables using above list
        total = sum(totals)

        l = len(totals) // 2
        lv = ldf_['confirmed'].values[-l:]
        # set individual bar lables using above list
        for i in range(l):
            p1 = ax.patches[i]
            p2 = ax.patches[l + i]
            # get_x pulls left or right; get_height pushes up or down
            ax.text(p1.get_x() + .08, p1.get_height() + p2.get_height() + .5, lv[i], fontsize=15, color='dimgrey')

        plt.tick_params(labelright=True)  # labeltop=True,
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()

    def fit(self, first_date=None, init_add=0.0, new_confirmed_threshold=100.0, range_append_nr_entires=40):

        # Only take the data as valid starting with first_date and prepend it with ramp up data.
        mortality_analysis = MortalityAnalysis(self.region, first_date=first_date, init_add=init_add, df = self.df)
        if first_date is not None:
            ldf = mortality_analysis.prepend_df[mortality_analysis.prepend_df.index >= first_date]
        else:
            ldf = mortality_analysis.prepend_df

        fit_df0 = ldf[['confirmed', 'new_confirmed']].reset_index(drop=True)
        # .reset_index(drop=True).reset_index(name='x')
        fit_df0.index.name = 'x'
        fit_df0 = fit_df0.reset_index().astype(np.float)
        fit_df0.index = ldf.index
        fit_df0['x'] = fit_df0['x'] + 1.0
        self.fit_df0 = fit_df0

        # Don't take the last two days for curve fitting
        fit_df = fit_df0.iloc[:-2].copy()
        self.fit_df = fit_df

        max_value = np.max(fit_df.confirmed)

        f1 = FitExp(fit_df.x, fit_df.confirmed, fit_df.new_confirmed, [10, 0.2])
        try:
            f1.fit().fit2()
        except:
            f1.f_integral.seor = np.inf
            f1.f_derivative.seor = np.inf
        self.f1 = f1
        f2 = FitSig(fit_df.x, fit_df.confirmed, fit_df.new_confirmed, [max_value * 3 / 2, 0.1, -10])
        f2.fit().fit2()
        self.f2 = f2

        # f1 = curve_fit(fitExp, fit_df.x, fit_df.confirmed, [10, 0.2], 'exp')
        # f2 = curve_fit(fitSig, fit_df.x, fit_df.confirmed, [max_value * 3 / 2, 0.2, -10], 'sigmoid')

        self.fit_               = None
        self.fit_choices        = None
        self.sorted_fit_choices = None

        if f1.seor() < f2.seor():
            self.fit_ = f1
        else:
            # f2 = FitSig   (fit_df.x, fit_df.confirmed, fit_df.new_confirmed, [max_value * 3 / 2, 0.2, -10])
            # f2.fit()

            f3 = FitSigExt(fit_df.x, fit_df.confirmed, fit_df.new_confirmed, [max_value * 3 / 2, 0.1, -10, 100])
            f3.fit()
            self.f3 = f3

            f2_p0 = np.array([f2.f_derivative.popt[0], f2.f_derivative.popt[1], f2.f_derivative.popt[1], f2.f_derivative.popt[2]])
            f4 = FitSigAsymmetric(fit_df.x, fit_df.confirmed, fit_df.new_confirmed, f2_p0)
            f4.fit()
            self.f4 = f4

            f3_p0 = np.array([f3.f_derivative.popt[0], f3.f_derivative.popt[1], f3.f_derivative.popt[1], f3.f_derivative.popt[2], f3.f_derivative.popt[3]])
            f5 = FitSigExtAsymmetric(fit_df.x, fit_df.confirmed, fit_df.new_confirmed, f3_p0)
            f5.fit()
            self.f5 = f5

            # f2   = curve_fit(fitSigDerivative, fit_df.x, fit_df.new_confirmed, [max_value * 3 / 2, 0.2, -10], 'sigmoid')
            # f2.fn_integral   = fitSig
            # f2.fn_derivative = fitSigDerivative
            #
            # p0 = np.array([f2.popt[0], f2.popt[1], f2.popt[1], f2.popt[2]])
            # f2_1 = curve_fit(fitSigAsymmetricDerivative, fit_df.x, fit_df.new_confirmed, p0, 'sigmoid+asymmetric')
            # f2_1.fn_integral   = fitSigAsymmetric
            # f2_1.fn_derivative = fitSigAsymmetricDerivative
            #
            # f2_2 = curve_fit_minimize(fitSigAsymmetricDerivative, fit_df.x, fit_df.new_confirmed, p0, 'sigmoid+asymmetric')
            # if f2_2.popt[0] < 10.0 or f2_2.popt[1] < 0.0001 or f2_2.popt[2] < 0.0001:
            #     bounds = ((10.0, None), (0.0001, None), (0.0001, None), (None, None))
            #     f2_2 = curve_fit_minimize(fitSigAsymmetricDerivative, fit_df.x, fit_df.new_confirmed, p0, 'sigmoid+asymmetric', bounds=bounds)
            # f2_2.fn_integral   = fitSigAsymmetric
            # f2_2.fn_derivative = fitSigAsymmetricDerivative
            #
            # f3   = curve_fit(fitSigExtDerivative, fit_df.x, fit_df.new_confirmed, [max_value * 3 / 2, 0.2, -10, 100], 'sigmoid+linear')
            # f3.fn_integral   = fitSigExt
            # f3.fn_derivative = fitSigExtDerivative
            #
            # p0 = np.array([f3.popt[0], f3.popt[1], f3.popt[1], f3.popt[2], f3.popt[3]])
            # f3_1 = curve_fit(fitSigExtAsymmetricDerivative, fit_df.x, fit_df.new_confirmed, p0, 'sigmoid+linear+asymmetric')
            # f3_1.fn_integral   = fitSigExtAsymmetric
            # f3_1.fn_derivative = fitSigExtAsymmetricDerivative
            #
            # f3_2 = curve_fit_minimize(fitSigExtAsymmetricDerivative, fit_df.x, fit_df.new_confirmed, p0, 'sigmoid+linear+asymmetric')
            # if f3_2.popt[0] < 10.0 or f3_2.popt[1] < 0.0001 or f3_2.popt[2] < 0.0001 or f3_2.popt[4] < 10.0:
            #     bounds = ((10.0, None), (0.0001, None), (0.0001, None), (None, None), (10.0, None))
            #     f3_2 = curve_fit_minimize(fitSigExtAsymmetricDerivative, fit_df.x, fit_df.new_confirmed, p0, 'sigmoid+linear+asymmetric', bounds=bounds)
            # f3_2.fn_integral   = fitSigExtAsymmetric
            # f3_2.fn_derivative = fitSigExtAsymmetricDerivative
            #
            # fit_choices = [f2, f2_1, f2_2, f3, f3_1, f3_2]
            # self.fit_choices = fit_choices.copy()
            # sorted_fit_choices = sorted(fit_choices, key=lambda x: x.seor)
            # self.sorted_fit_choices = sorted_fit_choices
            #
            # self.fit_ = sorted_fit_choices[0]

            fit_choices             = [f2, f3, f4, f5]
            self.fit_choices        = fit_choices.copy()
            sorted_fit_choices      = sorted(fit_choices, key=lambda x: x.seor())
            self.sorted_fit_choices = sorted_fit_choices
            self.fit_ = sorted_fit_choices[0]
            self.fit_.fit2()


        last_x = int(fit_df0.x.iloc[-1])
        last_day = fit_df0.index[-1]
        for i in range(1, range_append_nr_entires):
            x = last_x + i
            d = last_day + datetime.timedelta(days=i)
            fit_df0.loc[d] = [x, np.nan, np.nan]

        # # fitFunc = fitSig
        # proj = self.fit_.call_fn_integral(fit_df0.x)
        # fit_df0['fit'] = proj
        #
        # proj = self.fit_.call_fn_integral(fit_df.x)
        # growthRate = proj[-1] / proj[-2] - 1
        # self.growthRate = growthRate
        #
        # proj = self.fit_.call_fn_derivative(fit_df0.x)
        # fit_df0['fit_diff'] = proj

        proj = self.fit_.f_integral.call(fit_df0.x)
        fit_df0['fit'] = proj

        proj = self.fit_.f_integral.call(fit_df.x)
        growthRate = round(proj[-1] / proj[-2] - 1, 3)
        self.growthRate = growthRate

        proj = self.fit_.f_derivative.call(fit_df0.x)
        fit_df0['fit_diff'] = proj

        max_above_100_date = fit_df0[fit_df0['fit_diff'] > new_confirmed_threshold * 1.0].index.max()
        max_date = fit_df0.index.max()
        if max_above_100_date + datetime.timedelta(days=2) < max_date:
            max_above_100_date = max_above_100_date + datetime.timedelta(days=2)
        else:
            max_above_100_date = max_date
        self.max_above_100_date = max_above_100_date
        self.fit_df0 = fit_df0

    def fit_overview(self):
        return [item.f_derivative for item in self.fit_choices]

    def plot_with_fits(self, ax=None, restriction_start_date=None):
        if ax is None:
            fig = plt.figure(figsize=(32,8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1,1,1)
        # https://stackoverflow.com/questions/26752464/how-do-i-align-gridlines-for-two-y-axis-scales-using-matplotlib
        self.fit_df0[['confirmed']].plot(ax=ax, marker=mpl.path.Path.unit_circle(), markersize=5);
        self.fit_df0[['fit']].plot(ax=ax);
        # ax.set_ylim(-100,None)

        if restriction_start_date is not None:
            ax.axvline(restriction_start_date)

        ax2 = ax.twinx()
        self.fit_df0[['fit_diff']].plot(ax=ax2);
        self.fit_df0[['new_confirmed']].reset_index().plot.scatter(ax=ax2, x = 'index', y = 'new_confirmed', c='limegreen')
        # ax2.set_ylim(-100,None)

        # l = len(ax.get_yticks())
        # a1 = ax.get_yticks()[0]
        # e1 = ax.get_yticks()[-1]
        # a2 = ax2.get_yticks()[0]
        # e2 = ax2.get_yticks()[-1]
        # ax.set_yticks(np.linspace(a1, e1, l));
        # ax2.set_yticks(np.linspace(a2, e2, l));
        ax2.grid(None)
        itm = self.fit_df0.loc[self.max_above_100_date]
        print('{}; growth-rate: {}, date:{}, projected value: {}'.format(self.fit_, self.growthRate, itm.name, itm['fit_diff']))

    def calculate_R_estimates(self):
        self.prepareRdf()
        self.calculate_R_from_fit_diff()
        self.calc_R_from_lnqI()
        return self.Rdf

    def prepareRdf(self, column_name='new_confirmed', average_infectious_period=7.0):
        in_df = self.df
        rdf = pd.DataFrame(index=in_df.index)

        self.average_infectious_period = average_infectious_period
        self.gamma = 1 / average_infectious_period

        ws = int(average_infectious_period)
        if ws < average_infectious_period:
            ws += 1
        self.ws = ws

        rdf['x'] = np.arange(len(in_df)) * 1.0
        rdf['v'] = in_df[column_name]
        rdf['I_t'] = in_df[column_name].rolling(window=ws).sum()  # , min_periods=1

        rdf = rdf.iloc[:-2].copy()
        self.Rdf = rdf

    def calculate_R_from_fit_diff(self):

        fit_diff_column_name = 'fit_diff'
        rdf = self.fit_df0[[fit_diff_column_name]].copy()
        lds = rdf[fit_diff_column_name]
        rdf['v'] = lds
        rdf['I_t'] = lds.rolling(window=self.ws).sum()  # , min_periods=1
        rdf['qI'] = discrete_division(rdf['I_t'])
        rdf['gr_It'] = rdf['qI'] - 1.0

        # rdf['dI_t'] = discrete_diff(rdf['I_t'].values)
        # lda_gr_It = rdf['dI_t'].values[1:] / rdf['I_t'].values[:-1]
        # lda_gr_It = np.concatenate([np.array([np.nan]), lda_gr_It])
        # rdf['gr_It'] = lda_gr_It

        lda_R = np.maximum(1.0 + 1 / self.gamma * rdf['gr_It'], 0.0)
        rdf['R_t'] = lda_R

        self.Rdf['fit_R'] = rdf['R_t'].reindex(self.Rdf.index)

    def calc_R_from_lnqI(self):

        ldf_I = self.Rdf[['I_t']].copy()
        ldf_I['x'] = np.arange(len(ldf_I)) * 1.0
        ldf_I['qI'] = discrete_division(ldf_I['I_t'])
        ldf_I['lnqI'] = np.log(ldf_I['qI'])

        self.calc_R_from_lnqI_gp(ldf_I)
        self.calc_R_from_lnqI_KF(ldf_I)
        self.calc_R_from_lnqI_ll(ldf_I)

        ldf_R = ldf_I[['lnqI', 'gp_lnqI', 'kf_lnqI', 'll_lnqI']].copy()
        ldf_R.columns = ['R', 'gp_R', 'kf_R', 'll_R']
        ldf_R = np.exp(ldf_R) - 1

        ldf_R = np.maximum(1.0 + 1 / self.gamma * ldf_R, 0.0)
        ldf_R = ldf_R.reindex(self.Rdf.index)

        self.Rdf = pd.concat([self.Rdf, ldf_R], axis=1, sort=False)

        mean_R_columns = ['fit_R', 'gp_R', 'll_R']
        self.mean_R_columns = mean_R_columns
        self.Rdf['mean_R'] = self.Rdf[mean_R_columns].mean(axis=1)
        all_R_columns = ['R', 'fit_R', 'gp_R', 'kf_R', 'll_R', 'mean_R']
        self.all_R_columns = all_R_columns
        self.show_R_columns = all_R_columns[1:]

        self.Idf = ldf_I

    def calc_R_from_lnqI_gp(self, ldf_I):
        ldf_I['gp_lnqI'] = np.nan

        try:
            self.calc_R_from_lnqI_gp_gpflow(ldf_I)
        except Exception as e:
            warnings.warn("calc_R_from_lnqI_gp_gpflow failed")

            try:
                self.calc_R_from_lnqI_gp_gpy(ldf_I)
            except Exception as e:
                warnings.warn("calc_R_from_lnqI_gp_gpy failed")

    def calc_R_from_lnqI_gp_gpflow(self, ldf_I):
        ldf = ldf_I[np.isfinite(ldf_I.lnqI)]
        X = ldf.x.values.reshape(-1, 1)
        Y = ldf.lnqI.values.reshape(-1, 1)

        k1 = gpflow.kernels.RBF(variance=1.0, lengthscales=1.0)
        kernel = k1

        m = gpflow.models.GPR(data=(X, Y), kernel=kernel, noise_variance=1.0)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(m.training_loss, variables=m.trainable_variables, options=dict(disp=True), compile=False)  # 'Nelder-Mead', method='trust-exact', , maxiter=30
        ldf_I['gp_lnqI'] = m.predict_f(ldf_I.x.values.reshape(-1, 1))[0].numpy().reshape(-1)
        self.gp_m = m

    def calc_R_from_lnqI_gp_gpy(self, ldf_I):

        ldf = ldf_I[np.isfinite(ldf_I.lnqI)]
        X = ldf.x.values.reshape(-1, 1)
        Y = ldf.lnqI.values.reshape(-1, 1)

        k1 = k1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscales=1.)
        kernel = k1

        m = GPy.models.GPRegression(X, Y, kernel)
        m.optimize_restarts(num_restarts=10, robust=True)

        ldf_I['gp_lnqI'] = m.predict(ldf_I.x.values.reshape(-1, 1))[0].numpy().reshape(-1)
        self.gp_m = m

    def calc_R_from_lnqI_KF(self, ldf_I):
        lds_ = ldf_I['lnqI'].copy()
        lds  = lds_[np.isfinite(lds_)].copy()
        lds_.iloc[:] = np.nan

        kf = simdkalman.KalmanFilter(
            state_transition=np.array([[1, 1], [0, 1]]),
            process_noise=np.diag([0.1, 0.01]),
            observation_model=np.array([[1, 0]]),
            observation_noise=1.0)
        kf = kf.em(lds, n_iter=10)
        smoothed = kf.smooth(lds)
        ldf_I['kf_lnqI'] = np.nan
        v = smoothed.observations.mean

        lds.iloc[:]  = v
        lds_.loc[lds.index] = lds
        ldf_I.loc[:,'kf_lnqI'] = lds_.values

        self.kf = kf

    def calc_R_from_lnqI_ll(self, ldf_I):
        lds_ = ldf_I['lnqI'].copy()
        lds  = lds_[np.isfinite(lds_)].copy()
        lds_.iloc[:] = np.nan

        mod_ll = sm.tsa.UnobservedComponents(lds, 'local level')
        res_ll = mod_ll.fit(maxiter=200, disp=False)

        lds.iloc[:]  = res_ll.smoothed_state[0]
        lds_.loc[lds.index] = lds
        ldf_I.loc[:,'ll_lnqI'] = lds_.values

    def plot_R(self, ax=None, nr_days_back=30, plot_start_date=None):
        if ax is None:
            fig = plt.figure(figsize=(32,8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1,1,1)

        plot_R_columns = ['fit_R', 'll_R', 'mean_R']
        self.plot_R_columns = plot_R_columns
        if plot_start_date is None:
            self.Rdf[plot_R_columns].iloc[-nr_days_back:].plot(ax=ax)
        else:
            self.Rdf[plot_R_columns].loc[pd.to_datetime(plot_start_date):].plot(ax=ax)
        ax.axhline(1.0)

        return ax

    def R(self):
        return self.Rdf[self.show_R_columns].iloc[[-1]]


def get_country_overview():
    if augment_time_series_from_daily_snapshots_date_range is not None and len(augment_time_series_from_daily_snapshots_date_range) > 0:
        ldf_confirmed, ldf_recovered, ldf_death = augment_time_series_from_daily_snapshots(augment_time_series_from_daily_snapshots_date_range)
    else:
        ldf_confirmed, ldf_recovered, ldf_death = time_series_19_covid_confirmed.copy(), time_series_19_covid_recovered.copy(), time_series_19_covid_death.copy()

    ldf_recovered = pd.merge(ldf_confirmed[['Country/Region', 'Province/State']], ldf_recovered, how='left', on=['Country/Region', 'Province/State']).fillna(0.0)

    ldf_confirmed = ldf_confirmed[['Country/Region', columns[-1]]].groupby(['Country/Region']).sum()
    ldf_recovered = ldf_recovered[['Country/Region', columns[-1]]].groupby(['Country/Region']).sum()
    # ldf_recovered = ldf_confirmed.copy()
    ldf_death     = ldf_death[['Country/Region', columns[-1]]].groupby(['Country/Region']).sum()

    # return ldf_confirmed, ldf_recovered, ldf_death

    # confirmed_column_name = 'confirmed_' + str(pd.to_datetime(columns[-1]).date())
    confirmed_column_name = 'confirmed'
    ldf_confirmed.columns = [confirmed_column_name]
    # recovered_column_name = 'recovered_' + str(pd.to_datetime(columns[-1]).date())
    recovered_column_name = 'recovered'
    ldf_recovered.columns = [recovered_column_name]
    # ldf_recovered['recovered'] = -1
    # death_column_name     = 'death_' + str(pd.to_datetime(columns[-1]).date())
    death_column_name = 'death'
    ldf_death.columns = [death_column_name]

    ldf = pd.concat([ldf_confirmed, ldf_recovered, ldf_death], axis=1, sort=True)

    for idx, row in ldf.iterrows():
        if idx in override_xlsx.sheet_names:
            ldf_overrid = get_cases_by_region_override(region=idx)
            csse_covid_19_data_date = pd.to_datetime(columns[-1]).date()
            covid_manual_excel_date = ldf_overrid.iloc[-1].name - datetime.timedelta(days=1)
            # print(idx, csse_covid_19_data_date, covid_manual_excel_date, csse_covid_19_data_date == covid_manual_excel_date)
            if csse_covid_19_data_date == covid_manual_excel_date:
                ldf.loc[idx] = ldf_overrid.iloc[-1]

    ldf = ldf.astype(np.int)

    ldf['death_rate'] = ldf[death_column_name] / ldf[confirmed_column_name] * 100.0
    ldf['death_rate_'] = ldf[death_column_name] / (ldf[recovered_column_name] + ldf[death_column_name] + 1.0) * 100.0

    ldf.index.name = ldf.index.name + '_' + str(pd.to_datetime(columns[-1]).date())
    return ldf.sort_values(['death_rate'], ascending=False)


def prepend_fill(in_df, abs_column_name, delta_column_name, first_count, original_df):
    l = len(in_df) + 1
    daily_prepend_count = int(first_count // l)
    daily_prepend_count_fraction = first_count / l - daily_prepend_count
    prepend_count_fraction = 0.0
    for i in range(len(in_df)):
        in_df[delta_column_name].iloc[i] = daily_prepend_count
        delta = int(prepend_count_fraction // 1)
        if delta > 1.0:
            in_df[delta_column_name].iloc[i] += delta
            prepend_count_fraction -= delta
        prepend_count_fraction += daily_prepend_count_fraction

    in_df[abs_column_name] = in_df[delta_column_name].cumsum()

    last_value = in_df[abs_column_name].iloc[-1]
    original_df[delta_column_name].iloc[0] = first_count - last_value


def prepend(in_df, first_date=None, init_add=0, mult=1.0):
    if first_date is None:
        first_date = in_df.index[0]
    # in_df = in_df.loc[first_date:].copy()
    in_df = in_df.loc[first_date:].astype(np.int)

    in_df.loc[:, 'confirmed'] = (in_df['confirmed'] + init_add) * mult
    in_df['confirmed'] = in_df['confirmed'].astype(np.int)

    in_df.loc[:, 'new_confirmed'].iloc[0] = in_df['confirmed'].iloc[0]
    in_df.loc[:, 'new_confirmed'].iloc[1:] = in_df['confirmed'].iloc[1:].values - in_df['confirmed'].iloc[:-1].values
    # in_df['new_confirmed'] = in_df['new_confirmed'].astype(np.int)

    prepend_period_in_days = 20
    date_range_1_start = first_date + datetime.timedelta(days=-prepend_period_in_days)
    date_range_1_end = first_date + datetime.timedelta(days=-1)
    # print(date_range_1_start, date_range_1_end)
    date_range_2_start = first_date + datetime.timedelta(days=-(2 * prepend_period_in_days))
    date_range_2_end = first_date + datetime.timedelta(days=-prepend_period_in_days - 1)
    # print(date_range_2_start, date_range_2_end)

    prepend_dates = pd.date_range(start=date_range_1_start, end=date_range_1_end, freq='D')
    prepend_df = pd.DataFrame(np.zeros((len(prepend_dates), len(in_df.columns)), dtype='int'), index=prepend_dates,
                              columns=in_df.columns)
    for name in ['confirmed', 'recovered', 'death']:
        first_confirmed_count = in_df[name].iloc[0]
        last_value = prepend_fill(prepend_df, name, 'new_' + name, first_confirmed_count, in_df)

    in_df = pd.concat([prepend_df, in_df])
    prepend_dates = pd.date_range(start=date_range_2_start, end=date_range_2_end, freq='D')
    prepend_df = pd.DataFrame(np.zeros((len(prepend_dates), len(in_df.columns)), dtype='int'), index=prepend_dates,
                              columns=in_df.columns)
    for name in ['confirmed', 'recovered', 'death']:
        first_confirmed_count = in_df[name].iloc[0]
        last_value = prepend_fill(prepend_df, name, 'new_' + name, first_confirmed_count, in_df)

    return pd.concat([prepend_df, in_df])

# # ldf = get_cases_by_region(region='Italy').loc[pd.to_datetime('2020-02-21'):].copy()
# # ldf.loc[:,'confirmed'] = (ldf['confirmed'] + 100)*2.0
# # ldf.loc[:,'new_confirmed'].iloc[0] = ldf['confirmed'].iloc[0]
# # ldf.loc[:,'new_confirmed'].iloc[1:] = ldf['confirmed'].iloc[1:].values - ldf['confirmed'].iloc[:-1].values
# ldf = get_cases_by_region(region='Italy')
# # ldf = prepend(ldf, first_date=pd.to_datetime('2020-02-21'), init_add=100, mult=1.5)
# ldf = prepend(ldf, first_date=pd.to_datetime('2020-02-21'))
# ldf

# ldf = get_cases_by_region(region='Mainland China')
# # ldf = prepend(ldf, first_date=pd.to_datetime('2020-02-21'), init_add=100, mult=1.5)
# ldf = prepend(ldf)
# ldf

def distribute_across_cases_linear(in_df, dt, new_death, timeline_days=3 * 7):
    three_weeks_ago = dt + datetime.timedelta(days=-timeline_days)
    ldf = in_df[in_df.start_date >= three_weeks_ago]
    already_deaths = ldf.observed_death.sum()
    ldf = ldf.loc[ldf.observed_death == False, :]
    available_death_slots = len(ldf)
    if available_death_slots < new_death:
        raise Exception('available_death_slots < new_death')
    death_indices = np.random.choice(len(ldf), new_death, replace=False)
    death_indices = ldf.index[death_indices]

    ldf = in_df[in_df.start_date >= three_weeks_ago]
    added_death = (ldf.observed_death == True).sum() - already_deaths
    # print(new_death, added_death, already_deaths)

    return death_indices

gamma_loc   = 2.0
gamma_k     = 3.0
gamme_theta = 3.0

# gamma_loc   = 16.454713143887975
# gamma_k     = 5.103143892368536
# gamme_theta = 0.26394210847694694
# gamma_mean

duration_onset_death_mean = 17.8
duration_onset_death_02_5_pc = 16.9
duration_onset_death_97_5_pc = 19.2

def fit_measure(in_array):
    loc, k, theta = in_array[0], in_array[1], in_array[2]
    dist = stats.gamma(k, loc=loc, scale=theta)
    mean = float(dist.stats('m'))
    q_02_5 = dist.ppf(0.025)
    q_97_5 = dist.ppf(0.975)
    r = (duration_onset_death_mean - mean)**2 + (duration_onset_death_02_5_pc - q_02_5)**2 + (duration_onset_death_97_5_pc - q_97_5)**2
    return r

r = scipy.optimize.minimize(fit_measure, np.array([gamma_loc, gamma_k, gamme_theta]), method='Nelder-Mead', tol=0.1)
gamma_loc   = r.x[0]
gamma_k     = r.x[1]
gamme_theta = r.x[2]
gamma_dist = stats.gamma(gamma_k, loc=gamma_loc, scale=gamme_theta)
gamma_mean = float(gamma_dist.stats('m'))
gamma_q_02_5 = gamma_dist.ppf(0.025)
gamma_q_97_5 = gamma_dist.ppf(0.975)


def distribute_across_cases_gamma(in_df, dt, new_death, timeline_days=6 * 7, gamma_distribution_parameters=None):
    three_weeks_ago = dt + datetime.timedelta(days=-timeline_days)
    ldf = in_df[in_df.start_date >= three_weeks_ago]
    already_deaths = ldf.observed_death.sum()
    ldf = ldf.loc[ldf.observed_death == False, :]
    available_death_slots = len(ldf)
    if available_death_slots < new_death:
        raise Exception('available_death_slots < new_death')

    ds_age = (dt - ldf.start_date).dt.days
    k, loc, theta = gamma_k, gamma_loc, gamme_theta
    if gamma_distribution_parameters is not None:
        k, loc, theta = gamma_distribution_parameters['k'], gamma_distribution_parameters['loc'], gamma_distribution_parameters['theta']

    distribution = stats.gamma(k, loc=loc, scale=theta).pdf(ds_age)
    s = distribution.sum()
    if s > 0:
        distribution = distribution / s
    else:
        distribution = np.full_like(distribution, 1.0/len(distribution))

    try:
        death_indices = np.random.choice(len(ldf), new_death, replace=False, p=distribution)
    except:
        # print(dt, len(ldf), new_death, distribution)
        print('distribute_across_cases_gamma: using uniform distribution for date: {}'.format(dt))
        distribution = np.full_like(distribution, 1.0/len(distribution))
        death_indices = np.random.choice(len(ldf), new_death, replace=False, p=distribution)
        # raise
    death_indices = ldf.index[death_indices]

    ldf = in_df[in_df.start_date >= three_weeks_ago]
    added_death = (ldf.observed_death == True).sum() - already_deaths
    # print(new_death, added_death, already_deaths)

    return death_indices


def generate_life_lines(in_df, random_seed=None, gamma_distribution_parameters=None):
    if random_seed is None:
        random_seed = 42  # np.random.RandomState(42)
    np.random.seed(random_seed)
    rdf = pd.DataFrame(columns=['start_date', 'end_date', 'observed_death'])
    end_date = in_df.index[-1] + datetime.timedelta(days=1)
    l = len(in_df)
    for i in range(l):
        dt = in_df.index[i]
        # end existing life-lines
        new_death = in_df['new_death'].iloc[i]
        if new_death > 0:
            # death_indices = distribute_across_cases_linear(rdf, dt, new_death, timeline_days=3*7)
            death_indices = distribute_across_cases_gamma(rdf, dt, new_death, gamma_distribution_parameters=gamma_distribution_parameters)
            rdf.loc[death_indices, 'observed_death'] = True
            rdf.loc[death_indices, 'end_date'] = dt

            # create new life-lines
        new_lifelines = in_df['new_confirmed'].iloc[i]
        line = [[dt, end_date, False]]
        ldf = pd.DataFrame(line * new_lifelines, columns=['start_date', 'end_date', 'observed_death'])
        rdf = pd.concat([rdf, ldf])
        rdf.reset_index(drop=True, inplace=True)

    rdf['day_count'] = (rdf.end_date - rdf.start_date).dt.days
    rdf['observed_death'] = rdf['observed_death'].astype(np.bool)

    return rdf


class MortalityAnalysis():
    def __init__(self, region, first_date=None, init_add=0, mult=1.0, gamma_distribution_parameters=None, df=None):
        self.region = region
        self.first_date = first_date
        self.init_add = init_add
        self.mult = mult
        self.gamma_distribution_parameters = gamma_distribution_parameters
        if df is None:
            self.df = get_cases_by_region(region=region)
        else:
            self.df = df

        self.prepend_df = prepend(self.df, first_date=first_date, init_add=init_add, mult=mult)

    def fit(self):
        self.fit1()
        self.fit2()

    def fit1(self):
        self.calculate_delay_between_new_cases_and_death()
        delay_between_new_cases_and_death_timeshift = max(self.delay_between_new_cases_and_death_timeshift, 0.0)
        loc = min(max(delay_between_new_cases_and_death_timeshift - (gamma_mean - gamma_loc), 0.0), gamma_loc)
        if self.gamma_distribution_parameters is None:
            self.gamma_distribution_parameters = dict(loc=loc, k=gamma_k, theta=gamme_theta)
        else:
            self.gamma_distribution_parameters = self.gamma_distribution_parameters

        self.prepare_prediction()

        self.df_lifelines_individual = generate_life_lines(self.prepend_df, gamma_distribution_parameters=self.gamma_distribution_parameters)

    #         observed_death_by_day = self.df_lifelines_individual[['end_date', 'observed_death']].groupby(['end_date']).sum()
    #         observed_death_by_day['observed_death'] = observed_death_by_day['observed_death'].astype(np.int)
    #         self.observed_death_by_day = observed_death_by_day

    #         ldf = self.prepend_df.new_death - self.observed_death_by_day.observed_death
    #         if len(ldf[ldf > 0.0]) > 0:
    #             raise Exception('MortalityAnalysis: the death in df_lifelines_individual do not match the ones in prepend_df')

    def calculate_delay_between_new_cases_and_death(self):
        first_date = self.first_date
        if first_date is None:
            first_date = self.prepend_df.index[0]
        ldf = self.prepend_df[self.prepend_df.index >= first_date].copy()
        self.ll_df = ldf

        ll = LeadLagByShiftAndScale3(ldf.confirmed, ldf.death)
        self.ll = ll
        ll.fit()

        self.delay_between_new_cases_and_death_popt         = ll.shift_and_scale_popt
        self.delay_between_new_cases_and_death_cfr_estimate = ll.scale
        self.delay_between_new_cases_and_death_timeshift    = ll.shift

    def prepare_prediction(self, fit_column='confirmed'):
        ldf = self.prepend_df.copy()
        first_date = self.first_date
        if first_date is None:
            first_date = self.prepend_df.index[0]

        country_df = ldf[fit_column].reset_index(drop=True)
        # .reset_index(drop=True).reset_index(name='x')
        country_df.index.name = 'x'
        country_df = country_df.reset_index().astype(np.float)
        country_df.index = ldf.index
        country_df['x'] = country_df['x'] + 1.0

        fit_df = country_df[country_df.index >= first_date].copy()
        popt, pcov, sqdiff, growthRate, proj, idx, fitFunc, label = find_best_fit(fit_df, fit_column=fit_column)
        self.prediction_fit_fitFunc    = fitFunc
        self.prediction_fit_popt       = popt
        self.prediction_fit_growthRate = growthRate
        self.prediction_fit_label      = label

        last_x = int(country_df.x.iloc[-1])
        last_day = country_df.index[-1]
        for i in range(1, 40):
            x = last_x + i
            d = last_day + datetime.timedelta(days=i)
            country_df.loc[d] = [x, np.nan]

        # fitFunc = fitSig
        proj = fitFunc(country_df.x, *popt)
        label_fit = label + '_fit'
        self.prediction_fit_label_fit = label_fit
        country_df[label_fit] = proj


        vs = np.concatenate([np.array([0.0]), country_df[label_fit].values[1:] - country_df[label_fit].values[:-1]])
        label_fit_diff = label + '_fit_diff'
        self.prediction_fit_label_fit_diff = label_fit_diff
        country_df[label_fit_diff] = vs

        self.prediction_fit_df = country_df

    def fit2(self):
        kmf_ = lifelines.KaplanMeierFitter()
        kmf_.fit(self.df_lifelines_individual.day_count, self.df_lifelines_individual.observed_death, label='kmf_')
        self.kmf = kmf_

        # wbf_ = lifelines.WeibullFitter()
        # wbf_.fit(self.df_lifelines_individual.day_count, self.df_lifelines_individual.observed_death, label='wbf_')
        # self.wbf = wbf_
        #
        # exf_ = lifelines.ExponentialFitter()
        # exf_.fit(self.df_lifelines_individual.day_count, self.df_lifelines_individual.observed_death, label='exf_')
        # self.exf = exf_

    def plot(self):
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.subplot(111)
        ax.set_ylim([0.0, 1.0])

        self.kmf.plot(ax=ax)
        # self.wbf.plot_survival_function(ax=ax)
        # self.exf.plot_survival_function(ax=ax)

    def death_rate(self):
        mean = np.round(float(1 - self.kmf.survival_function_.iloc[-1].values) * 100, 2)
        lower = np.round(float(1 - self.kmf.confidence_interval_.iloc[-1, 1]) * 100, 2)
        upper = np.round(float(1 - self.kmf.confidence_interval_.iloc[-1, 0]) * 100, 2)
        delay_between_new_cases_and_death_cfr_estimate = np.round(self.delay_between_new_cases_and_death_cfr_estimate * 100, 2)
        return (mean, lower, upper, delay_between_new_cases_and_death_cfr_estimate, self.delay_between_new_cases_and_death_timeshift)

    def print_death_rate(self):
        return 'CFR via Survival analysis: {} (lower: {}, upper:{}), CFR via shift and scale: {} (time delay between infection and death: {:.2f} days)'.format(*self.death_rate())

    def project_death_and_hospitalization(self):
        death_rate     = float(self.death_rate()[0] / 100.0)
        delta_days     = int(np.round(self.delay_between_new_cases_and_death_timeshift, 0))
        self.delta_days = delta_days
        dt = self.df.iloc[-1].name
        today_idx = int(np.argwhere(self.prediction_fit_df.index == dt))
        # today_idx = len(self.df) - 1
        self.today_idx = today_idx

        today_death_idx = today_idx - delta_days
        self.today_death_idx = today_death_idx

        window_size = 3 * 7
        v = self.prediction_fit_df[self.prediction_fit_label_fit]
        self.v = v
        v1 = v[today_death_idx : today_death_idx + window_size]
        self.v1 = v1
        v2 = v[today_death_idx - window_size : today_death_idx]
        self.v2 = v2
        vd = (v1.values - v2.values)
        self.vd = vd

        expected_death = self.prepend_df['confirmed'].iloc[-1] * death_rate
        today_death    = self.df['death'].iloc[-1]
        delta_death    = expected_death - today_death

        # delta_death_across_days = delta_death / delta_days
        delta_death_2  = np.max(self.vd) * death_rate
        delta_death_across_days =  delta_death_2 / window_size

        proportion_of_ventilator_patient_dies = 0.5
        # p1 =  proportion_of_ventilator_patient_dies * p => p = p1/proportion_of_ventilator_patient_dies
        p1 = death_rate
        p  = p1/proportion_of_ventilator_patient_dies
        # p = p1 + p2 => p2 = p - p1 : proportion of needing ventilator and survives
        p2 = p - p1

        dD = 17.8 - 9 # for patients who die the duration they need a ventilator. At day 9 of symptom onset they go into ICU. At day 17.8 they die.
        dS = dD + 10 #  for patients who need a ventilator, but who survive

        # Number of ventilators needed is N * p1 * dD + N * p2 * dS
        # N is the daily number of discovered cases
        N = np.max(self.vd) / window_size

        required_ventilator_capacity =  N * p1 * dD + N * p2 * dS
        # required_ventilator_capacity = delta_death / proportion_of_ventilator_patient_dies
        # required_ventilator_capacity = delta_death_2 / proportion_of_ventilator_patient_dies

        return pd.DataFrame([[expected_death, today_death, delta_death, delta_death_2, delta_death_across_days, window_size, required_ventilator_capacity]], columns=['expected_death', 'today_death', 'delta_death', 'expected_death_2', 'delta_death_across_days', 'delta_days', 'required_ventilator_capacity']).round(0)

    def plot_infection_and_death_curves(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(32,8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1,1,1)

        first_date = self.first_date
        if first_date is None:
            first_date = self.prepend_df.index[0]
        ldf = self.prepend_df[self.prepend_df.index >= first_date].copy()

        ll = LeadLagByShiftAndScale1(ldf.confirmed, ldf.death)
        ll.fit()

        ll.plot_lead_lag(ax=ax)

US_states1 = [
    'District of Columbia',
    'Guam',
    'Puerto Rico',
]

US_states0 = [
    'Alabama',
    'Alaska',
    'Arizona',
    'Arkansas',
    'California',
    'Colorado',
    'Connecticut',
    'Delaware',
    'Florida',
    'Georgia',
    'Hawaii',
    'Idaho',
    'Illinois'
    'Indiana',
    'Iowa',
    'Kansas',
    'Kentucky',
    'Louisiana',
    'Maine',
    'Maryland',
    'Massachusetts',
    'Michigan',
    'Minnesota',
    'Mississippi',
    'Missouri',
    'Montana',
    'Nebraska',
    'Nevada',
    'New Hampshire',
    'New Jersey',
    'New Mexico',
    'New York',
    'North Carolina',
    'North Dakota',
    'Ohio',
    'Oklahoma',
    'Oregon',
    'Pennsylvania',
    'Rhode Island',
    'South Carolina',
    'South Dakota',
    'Tennessee',
    'Texas',
    'Utah',
    'Vermont',
    'Virginia',
    'Washington',
    'West Virginia',
    'Wisconsin',
    'Wyoming',
]

US_states = US_states0 + US_states1

def get_us_data_for_time_series_(input_df):
    ldf1 = input_df[(input_df['Country/Region'] == 'US') & input_df['Province/State'].str.contains(r'^.*, .*$')]
    ldf1 = ldf1.loc[:,:'3/9/20']
    ldf2 = input_df[(input_df['Country/Region'] == 'US') & input_df['Province/State'].isin(US_states)]
    ldf2 = ldf2.loc[:,'3/10/20':]

    lds = pd.concat([ldf1.sum(), ldf2.sum()])

    ldf = pd.DataFrame(columns=(['Country/Region'] + list(columns)))
    lds = (['US'] + list(lds[columns].values))
    ldf.loc[0] = lds
    return ldf



def find_best_fit(country_df, fit_column='confirmed'):

    daynr = country_df.x
    values = country_df[fit_column]

    fitSets = [
        (0, [10, 0.2], fitExp, 'exp'),
        (1, [max(values) * 3 / 2, 0.2, -10], fitSig, 'sigmoid'),
        (2, [max(values) * 3 / 2, 0.2, -10, 100], fitSigExt, 'sigmoid+linear')
    ]

    n = len(values)

    bestSeor, bestIndex, bestPopt, bestPcov = sys.float_info.max, None, [], []
    sndbestSeor, sndbestIndex, sndbestPopt, sndbestPcov = sys.float_info.max, None, [], []

    for index, p0, fitFunc, _ in fitSets:
        try:


            # alpha=0.05
            try:
                # Standard error function first; gives best fit for last 2-3 weeks,
                # although fits for the early days are poorer
                popt, pcov = scipy.optimize.curve_fit(fitFunc, daynr, values, p0)
            except (RuntimeError, OverflowError, TypeError) as e:
                # Modified poisson error with x^4. Deal with outliers for the last few
                # days, which cause the fitting to fail with the standard error function.
                popt, pcov = scipy.optimize.curve_fit(fitFunc, daynr, values, p0, sigma=[math.pow(theY + 1.0, 1 / 4.0) for theY in values], absolute_sigma=True)

            # Small constant relative error: alpha*x. Performs worse for last 20 days
            # popt, pcov = curve_fit(f, x, yd, p0, sigma=[alpha*theY+1 for theY in yd], absolute_sigma=True)
            # Possion error: sqrt(y), inherent for all rate counting applications. Performs worse for last 20 days
            # popt, pcov = curve_fit(f, x, yd, p0, sigma=[math.sqrt(theY+1) for theY in yd], absolute_sigma=True)
            # Combined constant relative and poisson. Performs worse for last 20 days
            # popt, pcov = curve_fit(f, x, yd, p0, sigma=[math.sqrt(theY+alpha*alpha*theY*theY+1) for theY in yd], absolute_sigma=True)

            # r2         = 1.0-(sum((yd-f(x,*(popt)))**2)/((n-1.0)*np.var(yd,ddof=1)))
            # pseudo-R2 for nonlinear fits, from https://stackoverflow.com/a/14530853
            # Better use standard error of the estimate for a non-linear fit. Lower values are better
            seor = math.sqrt(sum((values - fitFunc(daynr, *(popt))) ** 2) / (n - len(popt)))

            if abs(popt[1] > 2):  # PRIOR: we know the exponent is fairly small, use this to ignore absurd fits
                continue
            if len(popt) == 4 and (popt[3] > popt[0]):  # PRIOR: we know the plateau should be smaller than the peak, ignore absurd fits
                continue
            if  len(popt) == 4 and (popt[1] * popt[3] < 0.0): # PRIOR: we know that the steady-state-rate (= b * n) cannot be negative
                continue

            # make this the new best result, if it exceeds the previous one by a threshold
            # added cache in case an intermediate result was somewhat better, but the new one isn't much better
            gamma = 1.1
            if seor * gamma < bestSeor and seor * gamma < sndbestSeor:
                bestSeor, bestIndex, bestPopt, bestPcov = seor, index, popt, pcov
                sndbestSeor = sys.float_info.max
            elif seor * gamma < bestSeor:
                if sndbestSeor < sys.float_info.max:
                    bestSeor, bestIndex, bestPopt, bestPcov = sndbestSeor, sndbestIndex, sndbestPopt, sndbestPcov
                sndbestSeor, sndbestIndex, sndbestPopt, sndbestPcov = seor, index, popt, pcov

        except (RuntimeError, OverflowError, TypeError) as e:
            continue

    seor, popt, pcov = bestSeor, bestPopt, bestPcov
    if bestIndex != None:
        index, p0, fitFunc, label = fitSets[bestIndex]
    else:
        index, p0, fitFunc, label = 0, [], None, 'None'

    # estimate error
    proj = fitFunc(daynr, *popt)

    # generate label for chart
    # equation=eqFormatter(popt)
    if (len(proj) >= 2 and proj[-2] != 0):
        growthRate = proj[-1] / proj[-2] - 1
    else:
        growthRate = 0

    return popt, pcov, seor, growthRate, proj, index, fitFunc, label


def fitCurve(country_df, fit_column='confirmed'):
    try:
        daynr = country_df.x
        values = country_df[fit_column]
        fitFunc = fitSig
        p0 = [values[-1], 0.1, -10]
        # fit curve
        popt, pcov = scipy.optimize.curve_fit(fitFunc, daynr, values, p0)

        # estimate error
        proj = fitFunc(daynr, *popt)
        sqdiff = np.sum((values - proj) ** 2)

        # generate label for chart
        # equation=eqFormatter(popt)
        if (len(proj) >= 2 and proj[-2] != 0):
            growthRate = proj[-1] / proj[-2] - 1
        else:
            growthRate = 0
        # fitLabel="%s\n%.1f%% daily growth" % (equation, 100*growthRate)
        return popt, pcov, sqdiff, growthRate, proj

    except (RuntimeError, TypeError) as e:
        raise e
        return [], [], sys.float_info.max, ""


def prepare_country_prediction(country_name, first_date, init_add=0.0, in_df=None, new_confirmed_threshold=100.0, fit_column='confirmed', range_append_nr_entires=40):
    if in_df is None:
        mortality_analysis = MortalityAnalysis(country_name, first_date=first_date, init_add=init_add)
        ldf = mortality_analysis.prepend_df[mortality_analysis.prepend_df.index >= first_date].copy()
    else:
        ldf = in_df.copy()

    country_df = ldf[fit_column].reset_index(drop=True)
    # .reset_index(drop=True).reset_index(name='x')
    country_df.index.name = 'x'
    country_df = country_df.reset_index().astype(np.float)
    country_df.index = ldf.index
    country_df['x'] = country_df['x'] + 1.0

    fit_df = country_df[country_df.index >= first_date].copy()
    popt, pcov, sqdiff, growthRate, proj, idx, fitFunc, label = find_best_fit(fit_df, fit_column=fit_column)

    last_x = int(country_df.x.iloc[-1])
    last_day = country_df.index[-1]
    for i in range(1, range_append_nr_entires):
        x = last_x + i
        d = last_day + datetime.timedelta(days=i)
        country_df.loc[d] = [x, np.nan]

    # fitFunc = fitSig
    proj = fitFunc(country_df.x, *popt)
    label_fit = label + '_fit'
    country_df[label_fit] = proj

    vs = np.concatenate([np.array([0.0]), country_df[label_fit].values[1:] - country_df[label_fit].values[:-1]])
    label_fit_diff = label + '_fit_diff'
    country_df[label_fit_diff] = vs

    country_df[fit_column + '_diff'] = np.concatenate([np.array([0.0]), country_df[fit_column].values[1:] - country_df[fit_column].values[:-1]])

    max_above_100_date = country_df[country_df[label_fit_diff] > new_confirmed_threshold * 1.0].index.max()
    max_above_100_date = max_above_100_date + datetime.timedelta(days=2)

    return country_df[country_df.index <= max_above_100_date], popt, pcov, sqdiff, growthRate, idx, label


def calculate_delay_between_new_cases_and_death(country_name, first_date, init_add=0.0, in_df=None):
    if in_df is None:
        mortality_analysis = MortalityAnalysis(country_name, first_date=first_date, init_add=init_add)
        if first_date is None:
            ldf = mortality_analysis.prepend_df.copy()
            first_date = mortality_analysis.prepend_df.index[0]
        else:
            ldf = mortality_analysis.prepend_df[mortality_analysis.prepend_df.index >= first_date].copy()
    else:
        ldf = in_df.copy()

    country_df = ldf[['confirmed', 'death']].reset_index(drop=True)
    # .reset_index(drop=True).reset_index(name='x')
    country_df.index.name = 'x'
    country_df = country_df.reset_index().astype(np.float)
    country_df.index = ldf.index
    country_df['x'] = country_df['x'] + 1.0

    fit_df = country_df[country_df.index >= first_date].copy()
    popt_confirmed, _, _, _, _ = fitCurve(fit_df, fit_column='confirmed')
    popt_death    , _, _, _, _ = fitCurve(fit_df, fit_column='death')

    x = country_df['x'].values
    extDayCount = 7
    t = np.linspace(x[0], x[-1] + extDayCount, 5 * (len(x) + extDayCount))
    death_predicted = fitSig(t, *popt_death)

    def fitdf(t, a, b):
        return a * fitSig(t - b, *(popt_confirmed))

    popt, pcov = scipy.optimize.curve_fit(fitdf, t, death_predicted, [0.05, 10])
    if popt[1] < 0:
        raise Exception('deaths must come after cases, ignore nonsensical fits')

    return popt


def get_rki(try_max=10):
    '''
    Downloads Robert Koch Institute data, separated by region (landkreis)

    Returns
    -------
    dataframe
        dataframe containing all the RKI data from arcgis.

    Parameters
    ----------
    try_max : int, optional
        Maximum number of tries for each query.
    '''

    landkreise_max = 413

    # Gets all unique landkreis_id from data
    url_id = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=0%3D0&objectIds=&time=&resultType=none&outFields=idLandkreis&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=true&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='
    url = urllib.request.urlopen(url_id)
    json_data = json.loads(url.read().decode())
    n_data = len(json_data['features'])
    unique_ids = [json_data['features'][i]['attributes']['IdLandkreis'] for i in range(n_data)]

    # If the number of landkreise is smaller than landkreise_max, uses local copy (query system can behave weirdly during updates)
    if n_data >= landkreise_max:

        print('Downloading {:d} unique Landkreise. May take a while.\n'.format(n_data))

        df_keys = ['Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall',
                   'AnzahlTodesfall', 'Meldedatum', 'NeuerFall', 'NeuGenesen', 'AnzahlGenesen']

        df = pd.DataFrame(columns=df_keys)

        # Fills DF with data from all landkreise
        for idlandkreis in unique_ids:

            url_str = 'https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0//query?where=IdLandkreis%3D' + idlandkreis + '&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall%2C+NeuGenesen%2C+AnzahlGenesen&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token='

            count_try = 0

            while count_try < try_max:
                try:
                    with urllib.request.urlopen(url_str) as url:
                        json_data = json.loads(url.read().decode())

                    n_data = len(json_data['features'])

                    if n_data > 5000:
                        raise ValueError('Query limit exceeded')

                    data_flat = [json_data['features'][i]['attributes'] for i in range(n_data)]

                    break

                except:
                    count_try += 1

            if count_try == try_max:
                raise ValueError('Maximum limit of tries exceeded.')

            df_temp = pd.DataFrame(data_flat)

            # Very inneficient, but it will do
            df = pd.concat([df, df_temp], ignore_index=True)

        df['date'] = df['Meldedatum'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1e3))

    else:

        print(
            "Warning: Query returned {:d} landkreise (out of {:d}), likely being updated at the moment. Using fallback (outdated) copy.".format(
                n_data, landkreise_max))
        this_dir = os.path.dirname(__file__)
        df = pd.read_csv(this_dir + "/../data/rki_fallback.csv", sep=",")
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

    return df


def fitExp(t, a, b):
    return a * np.exp(b * t)

def fitExpDerivative(x, a, b):
    return a * b * np.exp(b*x)

def fitSig(t, a, b, c):
    return a / (1.0 + np.exp(-b * t - c))

# Derivative of the sigmoid fit function
def fitSigDerivative(t, a, b, c):
    s=fitSig(t,1,b,c)
    return a*b*s*(1-s)

def fitSigAsymmetric(t, a, b1, b2, c):
    ti = b1*t+c
    negative  = (-np.sign(ti) + np.abs(np.sign(ti)))/2.0
    positive  = (np.sign(ti)  + np.abs(np.sign(ti)))/2.0

    b = negative * b1 + positive * b2

    c1 = c
    c2 = b2 / b1 * c
    c = negative * c1 + positive * c2

    a1 = a
    a2 = b1 / b2 * a
    a = negative * a1 + positive * a2

    dt = positive * (a2 - a1)/2.0

    return fitSig(t, a, b, c) - dt

# -b*t -c == 0 => t = -c/b
# b1*t+c = b2*t+c2 at -c/b => c2=b2*c/b1
def fitSigAsymmetricDerivative(t, a, b1, b2, c):
    ti = b1*t+c
    negative  = (-np.sign(ti) + np.abs(np.sign(ti)))/2.0
    positive  = (np.sign(ti)  + np.abs(np.sign(ti)))/2.0

    b = negative * b1 + positive * b2

    c1 = c
    c2 = b2 / b1 * c
    c = negative * c1 + positive * c2

    a1 = a
    a2 = b1 / b2 * a
    a = negative * a1 + positive * a2

    return fitSigDerivative(t, a, b, c)

def fitSigExt(t, a, b, c, n):
    exponent = b*t+c

    component1 = fitSig(t,a,b,c)

    component2 = np.zeros_like(exponent)
    selector = exponent > 20.0
    component2[selector] = exponent[selector]
    component2[~selector] = np.log(1 + np.exp(exponent[~selector]))
    component2 = n * component2

    return component1 + component2

def fitSigExtDerivative(t, a, b, c, n):
    s=fitSig(t,1,b,c)
    return a*b*s*(1-s) + fitSig(t, n*b, b,c)


def fitSigExtAsymmetric(t, a, b1, b2, c, n):
    ti = b1 * t + c
    negative = (-np.sign(ti) + np.abs(np.sign(ti))) / 2.0
    positive = (np.sign(ti) + np.abs(np.sign(ti))) / 2.0

    b = negative * b1 + positive * b2

    c1 = c
    c2 = b2 / b1 * c
    c = negative * c1 + positive * c2

    a1 = a
    a2 = (a * b1 / 4 + b1 * n / 2 - b2 * n / 2) * 4 / b2
    a = negative * a1 + positive * a2

    dt = positive * (a2 - a1) / 2.0

    return fitSigExt(t, a, b, c, n) - dt

def fitSigExtAsymmetricDerivative(t, a, b1, b2, c, n):
    ti = b1*t+c
    negative  = (-np.sign(ti) + np.abs(np.sign(ti)))/2.0
    positive  = (np.sign(ti)  + np.abs(np.sign(ti)))/2.0

    b = negative * b1 + positive * b2

    c1 = c
    c2 = b2 / b1 * c
    c = negative * c1 + positive * c2

    a1 = a
    a2 = (a * b1 / 4 + b1 * n / 2 - b2 * n / 2) * 4 / b2
    a = negative * a1 + positive * a2

    return fitSigExtDerivative(t, a, b, c, n)


def fn_minimize(fitFunc, x, y):
    def f(params):
        return np.sum((y - fitFunc(x, *params)) ** 2)

    return f

class FitResult():

    def __init__(self, fitFunc, seor, popt, pcov, label):
        self.fitFunc = fitFunc
        self.seor    = seor
        self.popt    = popt
        self.pcov    = pcov
        self.label   = label

    def call(self, x):
        return self.fitFunc(x, *self.popt)

    def __str__(self):
        return '{}: seor: {}, popt: {}'.format(self.label, self.seor, self.popt)

    def __repr__(self):
        return self.__str__()

def curve_fit(fitFunc, x, y, p0, label, bounds=None):
    try:

        bad = False
        popt = None
        if bounds is not None:
            try:
                popt, pcov = scipy.optimize.curve_fit(fitFunc, x, y, p0)
            except:
                warnings.warn('Exception in curve_fit 1')
                popt = p0
                bad = True
        else:
            popt, pcov = scipy.optimize.curve_fit(fitFunc, x, y, p0)

        if bounds is not None:
            for i, b in enumerate(bounds):
                if not (b[0] <= popt[i]  and popt[i] <= b[1]):
                    bad = True
                    break

            if bad:
                bl = [i[0] for i in bounds]
                bu = [i[1] for i in bounds]
                bounds_ = (bl, bu)
                popt, pcov = scipy.optimize.curve_fit(fitFunc, x, y, p0, bounds=bounds_)
                popt, pcov = scipy.optimize.curve_fit(fitFunc, x, y, popt)

        seor = math.sqrt(sum((y - fitFunc(x, *(popt))) ** 2) / (len(x) - len(popt)))
    except Exception as e:
        warnings.warn('Exception in curve_fit: ' + str(e) + '/ {}'.format(popt))
        pcov = None
        seor = np.inf

    return FitResult(fitFunc, seor, popt, pcov, label)

def curve_fit_minimize(fitFunc, x, y, p0, label, bounds=None):
    lf = fn_minimize(fitFunc, x, y)
    res = scipy.optimize.minimize(lf, p0, method='L-BFGS-B', bounds=bounds, options = dict(maxiter=100))  #
    if not res.success:
        return FitResult(fitFunc, np.inf, None, None, label)

    popt = res.x
    seor = math.sqrt(sum((y - fitFunc(x, *(popt))) ** 2) / (len(x) - len(popt)))
    return FitResult(fitFunc, seor, popt, None, label)

class CurveFit():

    def __init__(self, x, y, dy, p0, label):
        self.x = x
        self.y = y
        self.dy = dy
        self.p0 = p0
        self.label = label
        self.f_integral   = FitResult(None, np.inf, None, None, None)
        self.f_derivative = FitResult(None, np.inf, None, None, None)

    def predict_y(self, x):
        return self.f_integral.call(x)

    def predict_dy(self, x):
        return self.f_derivative.call(x)

    def seor(self):
        return self.f_derivative.seor

    def __str__(self):
        return '{}: seor: {}'.format(self.label, self.seor())

    def __repr__(self):
        return self.__str__()

class CurveFitWithConstraints(CurveFit):

    def __init__(self, x, y, dy, p0, label, bounds):
        super().__init__(x, y, dy, p0, label)
        self.bounds = bounds

    def fit_(self, fitFunc, ytarget, p0):
        f1 = curve_fit(fitFunc, self.x, ytarget, p0, self.label)
        if f1.seor == np.inf or not self.check_constraints_ok(f1):
            f1.seor = np.inf

        f2 = curve_fit_minimize(fitFunc, self.x, ytarget, p0, self.label)
        if f2.seor == np.inf or not self.check_constraints_ok(f2):
            f2 = curve_fit_minimize(fitFunc, self.x, ytarget, p0, self.label, bounds=self.bounds)

        if f1.seor < f2.seor:
            return f1
        else:
            return f2

class FitExp(CurveFit):

    def __init__(self, x, y, dy, p0):
        super().__init__(x, y, dy, p0, 'exp')

    def fit(self):
        f = curve_fit(fitExp, self.x, self.y, self.p0, self.label)
        self.f_integral   = f

        self.f_derivative = FitResult(fitExpDerivative, None, self.f_integral.popt, None, self.label)  # copy.copy(self.f_derivative)
        return self

    def fit2(self):
        self.f_derivative   = curve_fit(fitExpDerivative, self.x, self.dy, self.f_integral.popt, self.label)
        return self


class FitSig(CurveFit):

    def __init__(self, x, y, dy, p0):
        super().__init__(x, y, dy, p0, 'sigmoid')
        self.lower_a = 10.0
        self.lower_b = 0.0001
        self.bounds = [(1000.0, np.inf), (0.01, np.inf), (-np.inf, -0.1)]

    def check_constraints_ok(self, f):
        if f.popt is None:
            return False
        return not (f.popt[0] < self.lower_a or f.popt[1] < self.lower_b)

    def fit(self):
        f = curve_fit(fitSigDerivative, self.x, self.dy, self.p0, self.label, bounds=self.bounds)
        if not self.check_constraints_ok(f):
            f.seor = np.inf

        self.f_derivative = f

        self.f_integral   = FitResult(fitSig, None, self.f_derivative.popt, None, self.label)  # copy.copy(self.f_derivative)
        return self

    def fit2(self):
        self.f_integral   = curve_fit(fitSig, self.x, self.y, self.f_derivative.popt, self.label, bounds=self.bounds)
        return self

    def __str__(self):
        return super().__str__() + ', max asymptotic: {}'.format(self.f_integral.popt[0])

class FitSigAsymmetric(CurveFitWithConstraints):

    def __init__(self, x, y, dy, p0):
        self.lower_a  = 10.0
        self.lower_b1 = 0.0001
        self.lower_b2 = 0.0001
        super().__init__(x, y, dy, p0, 'sigmoid+asymmetric', ((self.lower_a, None), (self.lower_b1, None), (self.lower_b2, None), (None, None)))

    def check_constraints_ok(self, f):
        if f.popt is None:
            return False
        return not (f.popt[0] < self.lower_a or f.popt[1] < self.lower_b1 or f.popt[2] < self.lower_b2)

    def fit(self):
        self.f_derivative = self.fit_(fitSigAsymmetricDerivative, self.dy, self.p0)
        self.f_integral   = FitResult(fitSigAsymmetric, None, self.f_derivative.popt, None, self.label)
        return self

    def fit2(self):
        self.f_integral = self.fit_(fitSigAsymmetric, self.y, self.f_derivative.popt)
        return self

    def __str__(self):
        a = self.f_integral.popt[0]
        b1 = self.f_integral.popt[1]
        b2 = self.f_integral.popt[2]
        a1 = a
        a2 = b1 / b2 * a
        dt = (a2 - a1) / 2.0
        ma = a2 - dt
        return super().__str__() + ', max asymptotic: {}'.format(ma)

class FitSigExt(CurveFit):

    def __init__(self, x, y, dy, p0):
        super().__init__(x, y, dy, p0, 'sigmoid+linear')
        self.lower_a = 10.0
        self.lower_b = 0.0001
        self.lower_n = 1.0

    def check_constraints_ok(self, f):
        if f.popt is None:
            return False
        return not (f.popt[0] < self.lower_a or f.popt[1] < self.lower_b or f.popt[3] < self.lower_n)

    def fit(self):
        f = curve_fit(fitSigExtDerivative, self.x, self.dy, self.p0, self.label)
        if not self.check_constraints_ok(f):
            f.seor = np.inf

        self.f_derivative = f

        self.f_integral   = FitResult(fitSigExt, None, self.f_derivative.popt, None, self.label)  # copy.copy(self.f_derivative)
        return self

    def fit2(self):
        self.f_integral   = curve_fit(fitSigExt, self.x, self.y, self.f_derivative.popt, self.label)
        return self

    def __str__(self):
        return super().__str__() + ', max asymptotic: {}'.format(self.f_integral.popt[0])

class FitSigExtAsymmetric(CurveFitWithConstraints):

    def __init__(self, x, y, dy, p0):
        self.lower_a  = 10.0
        self.lower_b1 = 0.0001
        self.lower_b2 = 0.0001
        self.lower_n = 1.0
        super().__init__(x, y, dy, p0, 'sigmoid+asymmetric+linear', ((self.lower_a, None), (self.lower_b1, None), (self.lower_b2, None), (None, None), (self.lower_n, None)))

    def check_constraints_ok(self, f):
        if f.popt is None:
            return False
        return not (f.popt[0] < self.lower_a or f.popt[1] < self.lower_b1 or f.popt[2] < self.lower_b2 or f.popt[4] < self.lower_n)

    def fit(self):
        self.f_derivative = self.fit_(fitSigExtAsymmetricDerivative, self.dy, self.p0)
        self.f_integral   = FitResult(fitSigExtAsymmetric, None, self.f_derivative.popt, None, self.label)
        return self

    def fit2(self):
        self.f_integral = self.fit_(fitSigExtAsymmetric, self.y, self.f_derivative.popt)
        return self


def discrete_diff(in_da, first_value=np.nan):
    in_da = np.array(in_da)
    return np.concatenate([np.array([first_value]), in_da[1:] - in_da[:-1]])

def discrete_division(in_da, first_value=np.nan):
    in_da = np.array(in_da)
    return np.concatenate([np.array([first_value]), in_da[1:] / in_da[:-1]])


# https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv
# https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data
rki_df_url   = 'https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data'
rki_data_df = None
def get_rki_data():
    global rki_data_df

    if rki_data_df is not None:
        return rki_data_df

    rki_data_df = pd.read_csv(rki_df_url)# , encoding = "ISO-8859-1"
    return rki_data_df

def get_rki_df(state=None, county=None, time_anchor_column_name='Refdatum', replace_death=True): #, time_anchor_column_name='Meldedatum'
    ldf = get_rki_data()
    last_date = pd.to_datetime(ldf['Refdatum'].max()).tz_localize(None)
    # print(last_date)
    ldf = create_rki_df(ldf, state=state, county=county, time_anchor_column_name=time_anchor_column_name, last_date=last_date)
    if state is None and county is None and replace_death: # and time_anchor_column_name == 'Meldedatum'
        ldf_ = get_cases_by_region(region='Germany')
        ldf_ = ldf_.reindex(ldf.index)
        ldf['death'] = ldf_['death']
        ldf['new_death'] = ldf_['new_death']

    ldf = ldf.fillna(0.0).astype(np.int)

    return ldf

def timeline(in_df, state=None, county=None, time_anchor_column_name='Refdatum', count_column_name='AnzahlFall', last_date=None):
    ldf = in_df.copy()
    if state is not None:
        ldf = ldf[ldf['Bundesland'].str.contains(state)].copy()
    if county is not None:
        ldf = ldf[ldf['Landkreis'].str.contains(county)].copy()
    ldf[time_anchor_column_name] = pd.to_datetime(ldf[time_anchor_column_name]).dt.tz_localize(None)
    ldf = ldf.set_index(time_anchor_column_name)
    ldf.index.name = 'index'
    lds = ldf[count_column_name].copy()
    if last_date is not None:
        # print('ld: {}'.format(last_date))
        if last_date not in lds.index:
            lds.loc[last_date] = 0
        # print(lds.loc[last_date])
        # print(lds.tail())
    lds = lds.resample('D').sum()
    # print(lds.tail())
    return lds


def create_rki_df(in_df, state=None, county=None, time_anchor_column_name='Refdatum', last_date=None):
    lds_confirmed = timeline(in_df, state=state, county=county, time_anchor_column_name=time_anchor_column_name, count_column_name='AnzahlFall', last_date=last_date)
    lds_recovered = timeline(in_df, state=state, county=county, time_anchor_column_name=time_anchor_column_name, count_column_name='AnzahlGenesen', last_date=last_date)
    lds_death = timeline(in_df, state=state, county=county, time_anchor_column_name=time_anchor_column_name, count_column_name='AnzahlTodesfall', last_date=last_date)
    ldf = pd.DataFrame()
    ldf['confirmed'] = lds_confirmed.cumsum()
    ldf['recovered'] = lds_recovered.cumsum()
    ldf['death'] = lds_death.cumsum()

    ldf['new_confirmed'] = lds_confirmed
    ldf['new_recovered'] = lds_recovered
    ldf['new_death'] = lds_death
    return ldf

austria_df_url = 'https://opendata.arcgis.com/datasets/123014e4ac74408b970dd1eb060f9cf0_3.csv'
austria_data_df = None
def get_austria_df():
    global austria_data_df

    if austria_data_df is not None:
        return austria_data_df

    fname = austria_df_url
    alternative_austria_data = pd.read_csv(fname)
    alternative_austria_data['datum'] = pd.to_datetime(pd.to_datetime(alternative_austria_data.datum).dt.date)
    alternative_austria_data = alternative_austria_data[['datum', 'infizierte', 'genesene', 'verstorbene']].groupby(['datum']).sum()
    alternative_austria_data = alternative_austria_data.rename(
        columns={"infizierte": "confirmed", "verstorbene": "death", "genesene": "recovered"})

    for property in ['confirmed', 'recovered', 'death']:
        diff = alternative_austria_data[property].values[1:] - alternative_austria_data[property].values[:-1]
        alternative_austria_data['new_' + property] = np.concatenate([np.array([0]), diff])
    alternative_austria_data.index.name = 'index'
    alternative_austria_data = alternative_austria_data.fillna(0).astype(np.int)
    dt = pd.to_datetime(datetime.date.today()) - pd.DateOffset(1)
    alternative_austria_data = alternative_austria_data.fillna(0.0).astype(np.int)
    alternative_austria_data = alternative_austria_data.loc[:dt]
    return alternative_austria_data


italy_df_url   = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
italy_data_df = None
def get_italy_df():
    global italy_data_df

    if italy_data_df is not None:
        return italy_data_df

    fname = italy_df_url
    alternative_italy_data = pd.read_csv(fname)
    dates = pd.to_datetime(pd.to_datetime(alternative_italy_data['data']).dt.date)
    alternative_italy_data = alternative_italy_data.rename(
        columns={"totale_casi": "confirmed", "deceduti": "death", "dimessi_guariti": "recovered"})
    alternative_italy_data = alternative_italy_data[['confirmed', 'recovered', 'death']].copy()
    for property in ['confirmed', 'recovered', 'death']:
        diff = alternative_italy_data[property].values[1:] - alternative_italy_data[property].values[:-1]
        alternative_italy_data['new_' + property] = np.concatenate([np.array([0]), diff])
    alternative_italy_data.index = dates
    alternative_italy_data = alternative_italy_data.fillna(0.0).astype(np.int)
    italy_data_df = alternative_italy_data
    return italy_data_df

spain_df_url   = 'https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/nacional_covid19.csv'
spain_data_df = None
def get_spain_df():
    global spain_data_df

    if spain_data_df is not None:
        return spain_data_df

    fname = spain_df_url
    alternative_spain_data = pd.read_csv(fname)
    alternative_spain_data['altas'] = alternative_spain_data['altas'].fillna(method='ffill')
    alternative_spain_data.fillna(0.0, inplace=True)
    print('read csv')
    dates = pd.to_datetime(pd.to_datetime(alternative_spain_data['fecha']).dt.date)
    # alternative_spain_data['casos_total'] = alternative_spain_data['casos_pcr'] + alternative_spain_data['casos_test_ac']
    alternative_spain_data = alternative_spain_data.rename(columns={"casos_pcr": "confirmed", "fallecimientos": "death", "altas": "recovered"})
    alternative_spain_data = alternative_spain_data[['confirmed', 'recovered', 'death']].copy()
    for property in ['confirmed', 'recovered', 'death']:
        diff = alternative_spain_data[property].values[1:] - alternative_spain_data[property].values[:-1]
        alternative_spain_data['new_' + property] = np.concatenate([np.array([0]), diff])
    alternative_spain_data.index = dates
    alternative_spain_data.index.name = 'index'
    alternative_spain_data = alternative_spain_data.fillna(0).astype(np.int)
    dt = pd.to_datetime(datetime.date.today()) - pd.DateOffset(1)
    alternative_spain_data = alternative_spain_data.fillna(0.0).astype(np.int)
    spain_data_df = alternative_spain_data.loc[:dt]
    return spain_data_df

france_df_url   = 'https://raw.githubusercontent.com/opencovid19-fr/data/master/dist/chiffres-cles.csv'
france_data_df = None
def get_france_df():
    global france_data_df

    # if france_data_df is not None:
    #     return france_data_df

    fname = france_df_url
    alternative_france_data = pd.read_csv(fname)
    alternative_france_data = alternative_france_data[(alternative_france_data['granularite'] == 'pays') & (alternative_france_data['source_type'] == 'ministere-sante')]
    dates = pd.to_datetime(pd.to_datetime(alternative_france_data['date']).dt.date)
    alternative_france_data = alternative_france_data.rename(columns={"cas_confirmes": "confirmed", "deces": "death", "gueris": "recovered"})
    alternative_france_data['death'] += alternative_france_data['deces_ehpad']
    alternative_france_data = alternative_france_data[['confirmed', 'recovered', 'death']].copy()
    for property in ['confirmed', 'recovered', 'death']:
        diff = alternative_france_data[property].values[1:] - alternative_france_data[property].values[:-1]
        alternative_france_data['new_' + property] = np.concatenate([np.array([0]), diff])
    alternative_france_data.index = dates

    dt = pd.to_datetime(datetime.date.today()) - pd.DateOffset(1)
    alternative_france_data = alternative_france_data[~pd.isnull(alternative_france_data.confirmed)]
    alternative_france_data = alternative_france_data.fillna(0.0).astype(np.int)
    france_data_df = alternative_france_data.loc[:dt]
    return france_data_df


class FitCascade():

    def __init__(self, total_cases_ds, fit_df=None):
        self.total_ds = pd.Series(total_cases_ds)
        self.delta_ds = pd.Series(discrete_diff(total_cases_ds), index=self.total_ds.index)
        self.total_ds = self.total_ds.iloc[1:]
        self.delta_ds = self.delta_ds.iloc[1:]
        self.fit_df = fit_df

    def fit(self, first_date=None, init_add=0.0, delta_threshold=100.0, range_append_nr_entires=40):

        if self.fit_df is None:
            # Only take the data as valid starting with first_date and prepend it with ramp up data.
            df = pd.DataFrame()
            df['confirmed']     = self.total_ds
            df['new_confirmed'] = self.delta_ds
            df['recovered']     = 0
            df['new_recovered'] = 0
            df['death']     = 0
            df['new_death'] = 0
            mortality_analysis = MortalityAnalysis('None', first_date=first_date, init_add=init_add, df = df)
            if first_date is not None:
                ldf = mortality_analysis.prepend_df[mortality_analysis.prepend_df.index >= first_date]
            else:
                ldf = mortality_analysis.prepend_df

            ldf = ldf.rename(columns={"confirmed": "total", "new_confirmed": "delta"})
            fit_df0 = ldf[['total', 'delta']].reset_index(drop=True)
            # .reset_index(drop=True).reset_index(name='x')
            fit_df0.index.name = 'x'
            fit_df0 = fit_df0.reset_index().astype(np.float)
            fit_df0.index = ldf.index
            fit_df0['x'] = fit_df0['x'] + 1.0

            # Don't take the last two days for curve fitting
            fit_df = fit_df0.iloc[:-2].copy()
        else:
            fit_df = self.fit_df
            fit_df0 = fit_df

        # min = fit_df0.x.min() - 1
        # fit_df0['x'] = fit_df0['x']- min
        # fit_df['x'] = fit_df['x'] - min

        self.fit_df = fit_df
        self.fit_df0 = fit_df0

        max_value = np.max(fit_df.total)

        f1 = FitExp(fit_df.x, fit_df.total, fit_df.delta, [10, 0.2])
        try:
            f1.fit().fit2()
        except:
            f1.f_integral.seor = np.inf
            f1.f_derivative.seor = np.inf
        self.f1 = f1
        f2_popt = [max_value * 3 / 2, 0.2, -10]
        f2 = FitSig(fit_df.x, fit_df.total, fit_df.delta, f2_popt)
        f2.fit().fit2()
        self.f2 = f2

        self.fit_               = None
        self.fit_choices        = None
        self.sorted_fit_choices = None

        # if f1.seor() < f2.seor():
        #     self.fit_ = f1
        # else:

        f3 = FitSigExt(fit_df.x, fit_df.total, fit_df.delta, [max_value * 3 / 2, 0.2, -10, 100])
        f3.fit()
        self.f3 = f3


        if f2.seor() == np.inf:
            f2_p0 = [max_value * 3 / 2, 0.2, 0.2, -10]
        else:
            f2_p0 = np.array([f2.f_derivative.popt[0], f2.f_derivative.popt[1], f2.f_derivative.popt[1], f2.f_derivative.popt[2]])
        f4 = FitSigAsymmetric(fit_df.x, fit_df.total, fit_df.delta, f2_p0)
        f4.fit()
        self.f4 = f4

        if f3.seor() == np.inf:
            f3_p0 = [max_value * 3 / 2, 0.2, 0.2, -10, 100]
        else:
            f3_p0 = np.array([f3.f_derivative.popt[0], f3.f_derivative.popt[1], f3.f_derivative.popt[1], f3.f_derivative.popt[2], f3.f_derivative.popt[3]])
        f5 = FitSigExtAsymmetric(fit_df.x, fit_df.total, fit_df.delta, f3_p0)
        f5.fit()
        self.f5 = f5

        fit_choices             = [f1, f2, f3, f4, f5]
        self.fit_choices        = fit_choices.copy()
        sorted_fit_choices      = sorted(fit_choices, key=lambda x: x.seor())
        self.sorted_fit_choices = sorted_fit_choices
        self.fit_ = sorted_fit_choices[0]
        self.fit_.fit2()


        last_x = int(fit_df0.x.iloc[-1])
        last_day = fit_df0.index[-1]
        for i in range(1, range_append_nr_entires):
            x = last_x + i
            d = last_day + datetime.timedelta(days=i)
            fit_df0.loc[d] = [x, np.nan, np.nan]

        proj = self.fit_.f_integral.call(fit_df0.x)
        fit_df0['fit'] = proj

        proj = self.fit_.f_integral.call(fit_df.x)
        growthRate = round(proj[-1] / proj[-2] - 1, 3)
        self.growthRate = growthRate

        proj = self.fit_.f_derivative.call(fit_df0.x)
        fit_df0['fit_diff'] = proj

        max_above_100_date = fit_df0[fit_df0['fit_diff'] > delta_threshold * 1.0].index.max()
        max_date = fit_df0.index.max()
        if max_above_100_date + datetime.timedelta(days=2) < max_date:
            max_above_100_date = max_above_100_date + datetime.timedelta(days=2)
        else:
            max_above_100_date = max_date
        self.max_above_100_date = max_above_100_date
        self.fit_df0 = fit_df0

    def fit_overview(self):
        return [item.f_derivative for item in self.fit_choices]

    def plot(self, ax=None, axv_date=None, total_column_name='confirmed', delta_column_name='new_confirmed'):
        if ax is None:
            fig = plt.figure(figsize=(32,8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1,1,1)

        # https://stackoverflow.com/questions/26752464/how-do-i-align-gridlines-for-two-y-axis-scales-using-matplotlib
        fit_column_name = total_column_name + '_fit'
        diff_fit_column_name = delta_column_name + '_fit'
        ldf = self.fit_df0.rename(columns={"total": total_column_name, "delta": delta_column_name, "fit": fit_column_name, "fit_diff": diff_fit_column_name})

        ldf[[total_column_name]].plot(ax=ax, marker=mpl.path.Path.unit_circle(), markersize=5);
        ldf[[fit_column_name]].plot(ax=ax);
        # ax.set_ylim(-100,None)

        if axv_date is not None:
            ax.axvline(axv_date)

        ax2 = ax.twinx()
        ldf[[diff_fit_column_name]].plot(ax=ax2);
        ldf[[delta_column_name]].reset_index().plot.scatter(ax=ax2, x = 'index', y = delta_column_name, c='limegreen')
        # ax2.set_ylim(-100,None)

        # l = len(ax.get_yticks())
        # a1 = ax.get_yticks()[0]
        # e1 = ax.get_yticks()[-1]
        # a2 = ax2.get_yticks()[0]
        # e2 = ax2.get_yticks()[-1]
        # ax.set_yticks(np.linspace(a1, e1, l));
        # ax2.set_yticks(np.linspace(a2, e2, l));
        ax2.grid(None)
        itm = ldf.loc[self.max_above_100_date]
        print('{}; growth-rate: {}, date:{}, projected value: {}'.format(self.fit_, self.growthRate, itm.name, itm[diff_fit_column_name]))


class LeadLagByShiftAndScaleBase():
    def __init__(self, leader_total_cases_ds, follower_total_cases_ds, first_date=None):
        self.first_date = first_date
        self.leader_total_cases_ds = pd.Series(leader_total_cases_ds)
        self.leader_delta_cases_ds = pd.Series(discrete_diff(leader_total_cases_ds), index=self.leader_total_cases_ds.index)
        self.leader_total_cases_ds = self.leader_total_cases_ds.iloc[1:]
        self.leader_delta_cases_ds = self.leader_delta_cases_ds.iloc[1:]

        self.follower_total_cases_ds = pd.Series(follower_total_cases_ds)
        self.follower_delta_cases_ds = pd.Series(discrete_diff(follower_total_cases_ds), index=self.follower_total_cases_ds.index)
        self.follower_total_cases_ds = self.follower_total_cases_ds.iloc[1:]
        self.follower_delta_cases_ds = self.follower_delta_cases_ds.iloc[1:]

        self.create_df_x(self.leader_total_cases_ds, self.follower_total_cases_ds)

        self.leader_fit_df0, self.leader_fit_df = self.create_fit_df(self.leader_total_cases_ds, self.leader_delta_cases_ds)
        self.follower_fit_df0, self.follower_fit_df = self.create_fit_df(self.follower_total_cases_ds, self.follower_delta_cases_ds)

        if (len(self.leader_total_cases_ds) != len(self.leader_delta_cases_ds)) or \
                (len(self.leader_total_cases_ds) != len(self.follower_total_cases_ds)) or \
                (len(self.leader_total_cases_ds) != len(self.follower_delta_cases_ds)):
            raise Exception('Data series lengths do not match!')

    def create_df_x(self, leader_total_cases_ds, follower_total_cases_ds):

        i1 = leader_total_cases_ds.index
        i2 = follower_total_cases_ds.index

        min_date = np.minimum(i1[0], i2[0])
        max_date = np.maximum(i1[-1], i2[-1])
        dr = pd.date_range(min_date, max_date)

        ldf_x = pd.DataFrame(dict(x=np.arange(len(dr))+1.0), index= dr)
        self.df_x = ldf_x

    def create_fit_df(self, ds_total, ds_delta):

        fit_df0 = self.df_x.reindex(ds_total.index)
        fit_df0['total'] = ds_total.values
        fit_df0['delta'] = ds_delta.values

        fit_df0 = fit_df0[~(pd.isnull(fit_df0.x) | pd.isnull(fit_df0.total) | pd.isnull(fit_df0.delta))]

        if self.first_date is not None:
            fit_df0 = fit_df0[fit_df0.index >= self.first_date].copy()


        # Don't take the last two days for curve fitting
        fit_df = fit_df0.iloc[:-2].copy()
        return fit_df0, fit_df

    def shift_and_scale(self, fn_leader_predict_dy, fn_follower_predict_dy):
        self.fn_leader_predict_dy = fn_leader_predict_dy
        self.fn_follower_predict_dy = fn_follower_predict_dy

        extDayCount = 7
        t = np.linspace(self.df_x.x[0], self.df_x.x[-1] + extDayCount, 5 * (len(self.df_x) + extDayCount))
        self.t = t
        lda_follower_fit = fn_follower_predict_dy(t)
        self.lda_follower_fit = lda_follower_fit


        def fitdf(t, a, b):
            return a * fn_leader_predict_dy(t - b)

        popt, pcov = scipy.optimize.curve_fit(fitdf, t, lda_follower_fit, [0.05, 10])
        self.shift_and_scale_popt         = popt
        self.scale = popt[0]
        self.shift    = popt[1]

        if popt[1] < 0:
            warnings.warn('deaths must come after cases, ignore nonsensical fits')


    def plot(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(32,8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1,1,1)

        t = self.df_x.x
        lda_follower_fit             = self.fn_follower_predict_dy(t)
        lda_transformed_leader_fit   = self.scale * self.fn_leader_predict_dy(t - self.shift)

        ldf = pd.DataFrame(index=self.leader_total_cases_ds.index)
        ldf['follower_fit']                  = lda_follower_fit
        ldf['shifted_and_scaled_leader_fit'] = lda_transformed_leader_fit
        self.fit_df0 = ldf

        ldf.plot(ax=ax)
        self.follower_fit_df[['delta']].reset_index().plot.scatter(ax=ax, x='index', y='delta', c='blue') # , c='limegreen'

        ldf = self.leader_fit_df[['x', 'delta']].copy()
        x0 = ldf.x[0]
        for index, row in ldf.iterrows():
            ldf.loc[index,'x'] = ldf.index[0] + pd.DateOffset(days=(row['x'] + self.shift - x0))

        ldf['delta'] = self.scale * ldf['delta']
        self.scaled_and_shifted_leader_df = ldf
        ldf.plot.scatter(ax=ax, x='x', y='delta', c='orange') #,, c=np.array([['orange']])


    def plot_lead_lag(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(32,8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1,1,1)

        self.leader_fit_df0.plot.scatter(ax=ax, x='x', y='delta', c='lightblue')
        ax.plot(self.leader_fit_df0.x, self.fn_leader_predict_dy(self.leader_fit_df0.x), c='lightblue')

        ax2 = ax.twinx()
        self.follower_fit_df0.plot.scatter(ax=ax2, x='x', y='delta', c='darkblue')
        ax2.plot(self.follower_fit_df0.x, self.fn_follower_predict_dy(self.follower_fit_df0.x), c='darkblue')
        ax2.grid(None)


class LeadLagByShiftAndScale1(LeadLagByShiftAndScaleBase):

    def __init__(self, leader_total_cases_ds, follower_total_cases_ds, first_date=None):
        super().__init__(leader_total_cases_ds, follower_total_cases_ds, first_date=first_date)

    def fit(self):

        self.fc_leader = FitCascade(self.leader_fit_df.total, fit_df=self.leader_fit_df)
        self.fc_leader.fit(first_date=self.first_date)

        self.fc_follower = FitCascade(self.follower_fit_df.total, fit_df=self.follower_fit_df)
        self.fc_follower.fit(first_date=self.first_date)

        self.shift_and_scale(self.fc_leader.fit_.predict_dy, self.fc_follower.fit_.predict_dy)

class LeadLagByShiftAndScale2(LeadLagByShiftAndScaleBase):

    def __init__(self, leader_total_cases_ds, follower_total_cases_ds, first_date=None):
        super().__init__(leader_total_cases_ds, follower_total_cases_ds, first_date=first_date)


    def fit(self):

        self.fc_leader = FitCascade(self.leader_fit_df.total, fit_df=self.leader_fit_df)
        self.fc_leader.fit(first_date=self.first_date)

        fitFunc = self.fc_leader.fit_.f_derivative.fitFunc
        p0      = self.fc_leader.fit_.f_derivative.popt.copy()
        label   = self.fc_leader.fit_.f_derivative.label

        follower_fit_result = curve_fit(fitFunc, self.follower_fit_df.x, self.follower_fit_df.delta, p0, label)
        self.follower_fit_result = follower_fit_result

        self.shift_and_scale(self.fc_leader.fit_.predict_dy, lambda t: follower_fit_result.fitFunc(t, *follower_fit_result.popt))

class LeadLagByShiftAndScale3(LeadLagByShiftAndScaleBase):

    def __init__(self, leader_total_cases_ds, follower_total_cases_ds, first_date=None):
        super().__init__(leader_total_cases_ds, follower_total_cases_ds, first_date=first_date)

    def fit(self):

        if self.first_date is not None:
            self.df_x = self.df_x[self.df_x.index >= self.first_date].copy()

        max_value = np.max(self.leader_fit_df['total'])
        fit_sig_leader = FitSig(self.leader_fit_df.x, self.leader_fit_df['total'], self.leader_fit_df['delta'], [max_value * 3 / 2, 0.2, -10])
        fit_sig_leader.fit()
        self.fit_sig_leader = fit_sig_leader

        max_value = np.max(self.follower_fit_df['total'])
        fit_sig_follower = FitSig(self.follower_fit_df.x, self.follower_fit_df['total'], self.follower_fit_df['delta'], [max_value * 3 / 2, 0.2, -10])
        fit_sig_follower.fit()
        self.fit_sig_follower = fit_sig_follower

        self.shift_and_scale(fit_sig_leader.predict_dy, fit_sig_follower.predict_dy)


