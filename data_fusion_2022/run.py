import sys
import pickle
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import gc
import utils


def prepro_trans(trs):
    trs.columns = ['bank_id', 'mcc', 'cur', 'rub', 'tm']
    trs['rub'] = trs['rub'].astype(np.int32)

    if len(trs) > 100:
        trs = trs[
            (trs["cur"] != -1) & (trs["tm"] >= "2021-01-25") & (trs["tm"] < "2021-08-01")].copy()
        gc.collect()

    codes = pd.read_csv('mcc_codes.csv')
    trs['rub'] = trs['rub']*trs['cur'].map({48: 1, 50: 75, 60: 85, -1: 1})
    trs = trs.merge(codes[['MCC', 'cats']], left_on='mcc',
                    right_on='MCC').drop(columns=['MCC', 'mcc'])
    del codes
    gc.collect()

    rub_intr = [-np.inf, -100000, -50000, -25000, -10000, -5000,
                -2000, -1000, -500, -300, -100, 0, 500, 1000, 2000, 4000, 5000, 10000, 20000, 50000, 100000, np.inf]
    trs['rub_intr'] = pd.cut(trs['rub'], bins=rub_intr,
                             labels=list(range(1, len(rub_intr))))
    trs['mean_rub'] = trs.groupby('rub_intr', observed=True)['rub'].transform(
        lambda x: np.round(x.mean())).astype(np.int32)

    trs.drop(columns=['cur', 'rub', 'rub_intr'], inplace=True)

    pop_cats = utils.load_pickle('pop_cats001.pickle')

    new_cats = (trs['mean_rub'] > 0)*1e4 + trs['cats']
    ispop = new_cats.isin(pop_cats)
    trs['ncats'] = new_cats*ispop

    trs.loc[(trs['mean_rub'] <= 0) & (~ispop), 'ncats'] = 50000
    trs.loc[(trs['mean_rub'] > 0) & (~ispop), 'ncats'] = 60000

    trs['cats'] = trs['ncats'].astype(np.int32).astype('category')
    trs.drop(columns='ncats', inplace=True)

    return trs


def prepro_clicks(ks):
    ks.columns = ['rtk_id', 'site', 'tm']

    if len(ks) > 100:
        ks = ks[(ks["tm"] >= "2021-01-25") & (ks["tm"] < "2021-08-01")
                ].copy()
        gc.collect()

    ks['uniq_h'] = ks['tm'].dt.dayofyear * 24 + ks['tm'].dt.hour

    pop_cats = utils.load_pickle('pop_sites0001.pickle')

    ks.loc[~ks['site'].isin(pop_cats), 'site'] = 5000

    return ks


def get_embed_trans(trs):
    # part1
    b_cat_mean = trs.pivot_table(
        index="bank_id",
        columns="cats",
        values="mean_rub",
        aggfunc="mean",
        fill_value=0,
        observed=True,
        sort=False,
    ).astype(np.float32).fillna(0)

    utils.flat_cols(b_cat_mean, "p1b_catmean")

    # part1.5
    b_cat_count = trs.pivot_table(
        index="bank_id",
        columns="cats",
        values="mean_rub",
        aggfunc="count",
        fill_value=0,
        observed=True,
        sort=False,
    ).astype(np.float32).fillna(0)

    b_cat_count = (
        b_cat_count.div(b_cat_count.sum(axis=1), axis="index").astype(
            np.float32).fillna(0)
    )
    utils.flat_cols(b_cat_count, "p1b_catcount")

    # part2
    b_hour = trs[trs["mean_rub"] < 0].pivot_table(
        index="bank_id",
        columns=trs["tm"].dt.hour,
        values="mean_rub",
        aggfunc="count",
        fill_value=0,
        observed=True,
        sort=False,
    )
    b_hour = (
        b_hour.div(b_hour.sum(axis=1), axis="index").astype(
            np.float32).fillna(0)
    )
    utils.flat_cols(b_hour, "p2b_hour")

    # part3
    b_days = trs.assign(d=trs["tm"].dt.dayofyear).groupby(
        'bank_id', observed=True, sort=False)['d'].agg(utils.not_active_days).to_frame()

    b_days.columns = ['p3b_days']
    b_days['p3b_num_active_days'] = b_days['p3b_days'].apply(
        lambda x: utils.max_active_days-len(x)).astype(np.float32).fillna(0)
    b_days['p3b_mean_intr_days'] = b_days['p3b_days'].apply(
        utils.mean_intr_days).astype(np.float32).fillna(0)

    b_all = pd.concat([b_cat_mean, b_cat_count, b_hour, b_days], axis=1)

    b_all = utils.full_cols(b_all, 'b_all_cols.pickle').reset_index()
    b_all.rename(columns={"bank_id": "bank"}, inplace=True)

    return b_all


def get_embed_clicks(ks):

    # part1
    part1 = ks.pivot_table(
        index="rtk_id",
        columns="site",
        values="uniq_h",
        aggfunc="nunique",
        fill_value=0,
        observed=True,
        sort=False,
    )

    part1 = part1.div(part1.sum(axis=1), axis="index").astype(
        np.float32).fillna(0)
    utils.flat_cols(part1, "p1r_sitecount")

    # part2
    part2 = ks.pivot_table(
        index="rtk_id",
        columns=ks["tm"].dt.hour,
        values="uniq_h",
        aggfunc="nunique",
        fill_value=0,
        observed=True,
        sort=False,
    )
    part2 = part2.div(part2.sum(axis=1), axis="index").astype(
        np.float32).fillna(0)
    utils.flat_cols(part2, "p2r_hour")

    # part3
    part3 = ks.assign(d=ks["tm"].dt.dayofyear).groupby(
        'rtk_id', observed=True, sort=False)['d'].agg(utils.not_active_days).to_frame()
    part3.columns = ['p3r_days']

    part3['p3r_num_active_days'] = part3['p3r_days'].apply(
        lambda x: utils.max_active_days-len(x)).astype(np.float32).fillna(0)
    part3['p3r_mean_intr_days'] = part3['p3r_days'].apply(
        utils.mean_intr_days).astype(np.float32).fillna(0)

    r_all = pd.concat([part1, part2, part3], axis=1)

    r_all = utils.full_cols(r_all, 'r_all_cols.pickle').reset_index()
    r_all.rename(columns={"rtk_id": "rtk"}, inplace=True)

    return r_all


def main():
    data, output_path = sys.argv[1:]

    trs = pd.read_csv(f'{data}/transactions.csv',
                      parse_dates=["transaction_dttm"], dtype={'mcc_code': np.int16, 'currency_rk': np.int8})

    trs = prepro_trans(trs)
    trs_emb = get_embed_trans(trs)
    del trs
    gc.collect()

    ks = pd.read_csv(f'{data}/clickstream.csv', parse_dates=["timestamp"], usecols=[
                     "timestamp", 'user_id', 'cat_id'], dtype={'cat_id': np.int16})
    ks = prepro_clicks(ks)
    ks_emb = get_embed_clicks(ks)
    del ks
    gc.collect()

    cb = CatBoostClassifier()
    cb.load_model('model.cbm')

    if len(trs_emb) < 10:
        preds = utils.predict(trs_emb, ks_emb, cb, zeros_prob=0.83)
    else:
        probs = utils.predict_probs(trs_emb, ks_emb, cb, batch=50, n=150)

        w = utils.calc_weights(probs, ks_emb['rtk'], k=0.5)
        w[0.] = 1.

        probs = probs.merge(w, left_on='rtk', right_index=True)
        probs['pred'] = probs['pred']*probs['weights']
        probs.loc[probs['rtk'] == 0., 'pred'] = 0.93 * \
            probs.loc[probs['rtk'] != 0., 'pred'].max()

        preds = utils.probs_to_preds(probs, trs_emb['bank'], n=100)

    subm = []
    subm.extend(preds.values)
    subm = np.array(subm, dtype=object)

    np.savez(output_path, subm)


if __name__ == "__main__":
    main()
