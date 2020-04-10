
import numpy as np, scipy, scipy.stats as stats, scipy.special, scipy.misc, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, xarray as xr
import matplotlib as mpl
import lifelines

import pymc3 as pm

import theano as thno
import theano.tensor as T

import datetime, time, math
from dateutil import relativedelta

from collections import OrderedDict
import sys

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

override_xlsx_name = 'covid-manual-excel.xlsx'
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
    r = region
    if isinstance(region, str):
        region = [region]

    return get_cases_by_selector(time_series_19_covid_confirmed['Country/Region'].isin(region), region=r)


class CasesByRegion():
    def __init__(self, region, df=None):
        if df is None:
            self.df = get_cases_by_region(region=region)
        else:
            self.df = df

    def tail(self):
        return self.df.tail()

    def plot_daily_stats(self, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(32, 8), dpi=80, facecolor='w', edgecolor='k')
            ax = plt.subplot(1, 1, 1)
        last_day = self.df.index[-1]
        # self.df[['new_confirmed', 'recovered', 'death']].loc[last_day + datetime.timedelta(days=-20):].plot.bar(ax=ax)
        ax = self.df[['new_confirmed', 'new_recovered', 'new_death']].loc[last_day + datetime.timedelta(days=-20):].plot.bar(ax=ax)
        plt.tick_params(labelright=True)  # labeltop=True,
        return ax

    def plot_daily_stacked(self):
        fig = plt.figure(figsize=(32, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = plt.subplot(1, 1, 1)
        last_day = self.df.index[-1]
        ldf_ = self.df
        ldf = ldf_[['confirmed', 'new_confirmed']].copy()
        ldf['confirmed'] = ldf['confirmed'] - ldf['new_confirmed']
        ldf[['confirmed', 'new_confirmed']].loc[last_day + datetime.timedelta(days=-20):].plot.bar(ax=ax, stacked=True)

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
    def __init__(self, region, first_date=None, init_add=0, mult=1.0, gamma_distribution_parameters=None):
        self.region = region
        self.first_date = first_date
        self.init_add = init_add
        self.mult = mult
        self.gamma_distribution_parameters = gamma_distribution_parameters
        self.df = get_cases_by_region(region=region)
        self.prepend_df = prepend(self.df, first_date=first_date, init_add=init_add, mult=mult)

        self.calculate_delay_between_new_cases_and_death()
        loc = max(self.delay_between_new_cases_and_death_timeshift - (gamma_mean - gamma_loc), 0.0)
        if gamma_distribution_parameters is None:
            self.gamma_distribution_parameters = dict(loc=loc, k=gamma_k, theta=gamme_theta)
        else:
            self.gamma_distribution_parameters =gamma_distribution_parameters

        self.df_lifelines_individual = generate_life_lines(self.prepend_df, gamma_distribution_parameters=self.gamma_distribution_parameters)

    #         observed_death_by_day = self.df_lifelines_individual[['end_date', 'observed_death']].groupby(['end_date']).sum()
    #         observed_death_by_day['observed_death'] = observed_death_by_day['observed_death'].astype(np.int)
    #         self.observed_death_by_day = observed_death_by_day

    #         ldf = self.prepend_df.new_death - self.observed_death_by_day.observed_death
    #         if len(ldf[ldf > 0.0]) > 0:
    #             raise Exception('MortalityAnalysis: the death in df_lifelines_individual do not match the ones in prepend_df')

    def calculate_delay_between_new_cases_and_death(self):
        ldf = self.prepend_df.copy()
        first_date = self.first_date
        if first_date is None:
            first_date = self.prepend_df.index[0]
        ldf = self.prepend_df[self.prepend_df.index >= first_date].copy()

        country_df = ldf[['confirmed', 'death']].reset_index(drop=True)
        # .reset_index(drop=True).reset_index(name='x')
        country_df.index.name = 'x'
        country_df = country_df.reset_index().astype(np.float)
        country_df.index = ldf.index
        country_df['x'] = country_df['x'] + 1.0

        fit_df = country_df[country_df.index >= first_date].copy()
        popt_confirmed, _, _, _, _ = fitCurve(fit_df, fit_column='confirmed')
        popt_death, _, _, _, _ = fitCurve(fit_df, fit_column='death')

        x = country_df['x'].values
        extDayCount = 7
        t = np.linspace(x[0], x[-1] + extDayCount, 5 * (len(x) + extDayCount))
        death_predicted = fitSig(t, *popt_death)

        def fitdf(t, a, b):
            return a * fitSig(t - b, *(popt_confirmed))

        popt, pcov = scipy.optimize.curve_fit(fitdf, t, death_predicted, [0.05, 10])
        if popt[1] < 0:
            raise Exception('deaths must come after cases, ignore nonsensical fits')

        self.delay_between_new_cases_and_death_popt = popt
        self.delay_between_new_cases_and_death_cfr_estimate = popt[0]
        self.delay_between_new_cases_and_death_timeshift    = popt[1]

    def fit(self):
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

    def project_death_and_hospitalization(self):
        death_rate     = self.death_rate()[0] / 100.0
        expected_death = self.prepend_df['confirmed'].iloc[-1] * death_rate
        today_death    = self.df['death'].iloc[-1]
        delta_death    = expected_death - today_death
        delta_days     = self.delay_between_new_cases_and_death_timeshift
        delta_death_across_days = delta_death / delta_days
        proportion_of_ventilator_patient_dies = 0.4
        required_ventilator_capacity = delta_death / proportion_of_ventilator_patient_dies
        return pd.DataFrame([[expected_death, today_death, delta_death, delta_death_across_days, delta_days, required_ventilator_capacity]], columns=['expected_death', 'today_death', 'delta_death', 'delta_death_across_days', 'delta_days', 'required_ventilator_capacity']).round(0)

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


def fitExp(t, a, b):
    return a * np.exp(b * t)


def fitSig(t, a, b, c):
    return a / (1.0 + np.exp(-b * t - c))

def fitSigExt(t, a, b, c, n):
    return fitSig(t,a,b,c) + n*np.log(1+np.exp(b*t+c))


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


def prepare_country_prediction(country_name, first_date, init_add=0.0, in_df=None, new_confirmed_threshold=100.0, fit_column='confirmed'):
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
    for i in range(1, 40):
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