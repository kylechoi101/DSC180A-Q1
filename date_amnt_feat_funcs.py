import pandas as pd
import numpy as np
from datetime import datetime

def create_date_feats(fit_df):
# create copy of data
    fit_add = fit_df.copy(deep=True)
    fit_add['posted_date'] = pd.to_datetime(fit_add['posted_date'])
    
# day of the week
    fit_add['day_of_week'] = fit_add['posted_date'].dt.strftime('%a')
    
# month
    fit_add['month'] = fit_add['posted_date'].dt.month_name()
    
# quarter
    fit_add['quarter'] = fit_add['posted_date'].dt.quarter
    q_map = {1: 'q1', 2: 'q2', 3: 'q3', 4: 'q4'}
    fit_add['quarter'] = fit_add['quarter'].map(q_map)
    
# year
    fit_add['year'] = fit_add['posted_date'].dt.year
    
# average time btwn transactions
    df = fit_add.copy()
    df = df.sort_values(['prism_consumer_id', 'posted_date'])
    
    df['days_since_prev'] = df.groupby('prism_consumer_id')['posted_date'].diff().dt.days.fillna(0)
    df['avg_days_between_txn'] = df.groupby('prism_consumer_id')['days_since_prev'].transform('mean')
    
    df = df.sort_index() # Restore original order
    fit_add = df
    
# rolling avg time btwn transactions (window = 5) --> can be helpful for determining financial stability
    df = fit_add.sort_values(['prism_consumer_id', 'posted_date']).copy()
    
    df['rolling_avg_days_between_txn'] = (
        df.groupby('prism_consumer_id')['days_since_prev']
          .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    )
    
    fit_add = df.sort_index() # restore to original order
    
# time since first transaction
    df = fit_add.sort_values(['prism_consumer_id', 'posted_date']).copy()
    df['posted_date'] = pd.to_datetime(df['posted_date'])
    
    first_txn = ( # find first transaction date per customer
        df.groupby('prism_account_id')['posted_date']
          .transform('min')
    )
    
    df['days_since_first_txn'] = (df['posted_date'] - first_txn).dt.days # compute days since first transaction
    
    fit_add = df.sort_index() # restore to original order
    return fit_add

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_amnt_feats(fit_df):
    if 'days_since_first_txn' not in fit_df.columns:
        fit_df = create_date_feats(fit_df) # run date func

# whole dollar amounts
    fit_df['whole_dollar'] = fit_df['amount'] % 1 == 0
        
# difference from median amount of transactions per month per customer
    fit_df['month_med_amnt'] = (fit_df.groupby(['prism_consumer_id', 'year', 'month'])['amount']
                                           .transform('median')
                                          )
    fit_df['month_med_amnt_diff'] = fit_df['amount']-fit_df['month_med_amnt']
    
# Standard deviation of amounts per consumer
    group_stats = fit_df.groupby('prism_consumer_id')['amount'].agg(['mean', 'std'])
    mean,std = group_stats['mean'],group_stats['std']
    fit_df['amnt_zscore'] = (fit_df['amount'] - mean) / std # compute z-score
    fit_df['amnt_zscore'] = fit_df['amnt_zscore'].fillna(0) # optional: fill NaN z-scores (e.g. if std = 0 or only one transaction)
    
# Log-transformed amount --> fix skewness of amounts
    fit_df['log_amnt'] = np.log1p(fit_df['amount'])
    
    return fit_df