import pandas as pd
from pandas import DataFrame


def from_dict_value_to_df(d: dict):
    """
    input = dictionary
    output = dataframe as part of all the values from the dictionary
    """
    df = pd.DataFrame()
    for v in d.values():
        df = df.append(v)
    return df


def simple_hot_encoding(df: DataFrame):
    # GENDER
    df['Gender'].replace(['Men', 'Women'],
                         [0, 1], inplace=True)
    gender_zip = zip(('Men', 'Women'), (0, 1))

    # USAGE
    df['Usage'].replace(['New', 'Almost New', 'Used', 'Very Used'],
                        [0, 1, 2, 3], inplace=True)
    usage_zip = zip(('New', 'Almost New', 'Used', 'Very Used'), (0, 1, 2, 3))

    # BRAND
    result = dict()
    for i in list(dict(df['Brand'].value_counts().loc[lambda x: x > 30]).keys()):
        result[i] = df[df['Brand'] == i]
    df_brands = from_dict_value_to_df(result)
    df_brands['Brand'] = df_brands['Brand'].str.lower()
    brands_zip = zip(tuple(list(df_brands['Brand'].unique())), tuple(range(len(list(df_brands['Brand'].unique())))))
    df_brands['Brand'].replace(list(df_brands['Brand'].unique()),
                               range(len(list(df_brands['Brand'].unique()))), inplace=True)
    df_brands = df_brands.loc[~(df_brands['Usage'].isin(['Underwear']))]

    return df_brands, dict(gender_zip), dict(usage_zip), dict(brands_zip)
