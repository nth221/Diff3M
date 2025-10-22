import pandas as pd
import numpy as np

def analysis_mimic(mimic_train_path, mimic_test_path):
    pd.set_option('display.max_columns', None)

    df = pd.read_csv(mimic_train_path)
    print(df.describe(include='all'))

    print('column dtype: ', df.dtypes)

    count = ((df['No Finding'].isna())).sum()
    print('No Finding == Nan: ', count)
    count = ((df['No Finding'] == 1)).sum()
    print('No Finding == 1: ', count)

    count = ((df['Support Devices'].isna())).sum()
    print('Support Devices == Nan: ', count)
    count = ((df['Support Devices'].isna() == False)).sum()
    print('Support Devices != Nan: ', count)

    count = ((df['Support Devices'].isna() == False) & (df['No Finding'] == 1)).sum()
    print('Support Devices != Nan and No Finding == 1: ', count)
    count = ((df['Support Devices'] == 1) & (df['No Finding'] == 1)).sum()
    print('Support Devices == 1 and No Finding == 1: ', count)
    count = ((df['Support Devices'] == 0) & (df['No Finding'] == 1)).sum()
    print('Support Devices == 0 and No Finding == 1: ', count)
    count = ((df['Support Devices'] == 0) & (df['No Finding'].isna())).sum()
    print('Support Devices == 0 and No Finding == Nan: ', count)
    count = ((df['Support Devices'] == -1) & (df['No Finding'] == 1)).sum()
    print('Support Devices == -1 and No Finding == 1: ', count)
    count = ((df['Support Devices'] == -1) & (df['No Finding'].isna())).sum()
    print('Support Devices == -1 and No Finding == Nan: ', count)
    count = ((df['Support Devices'].isna()) & (df['No Finding'] == 1)).sum()
    print('Support Devices == Nan and No Finding == 1: ', count)

    count = ((df['Support Devices'].isna() == False) & (df['No Finding'] == 1) & (df['BMI (kg/m2)'] == 0) & (df['Blood Pressure Max'] == 0)& (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] == 0) & (df['Weight (Lbs)'] == 0) ).sum()
    print('Support Devices != Nan and No Finding == 1 and EHR values are zero: ', count)
    count = ((df['BMI (kg/m2)'] == 0) & (df['Blood Pressure Max'] == 0) & (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] == 0) & (df['Weight (Lbs)'] == 0) ).sum()
    print('EHR values are zero: ', count)
    count = ((df['BMI (kg/m2)'] != 0) & (df['Blood Pressure Max'] != 0) & (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] != 0) & (df['Weight (Lbs)'] != 0) ).sum()
    print('EHR values are not zero: ', count)
    count = ((df['No Finding'] == 1) & (df['BMI (kg/m2)'] != 0) & (df['Blood Pressure Max'] != 0)& (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] != 0) & (df['Weight (Lbs)'] != 0) ).sum()
    print('No Finding == 1 and EHR values are not zero: ', count)

def make_mimic_1_5_dataset(mimic_train_path, mimic_test_path):
    # pd.set_option('display.max_columns', None)

    df = pd.read_csv(mimic_train_path)
    print('df original shape: ', df.shape)    
    
    df_filtered = df[(df['BMI (kg/m2)'] != 0) & (df['Blood Pressure Max'] != 0) & (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] != 0) & (df['Weight (Lbs)'] != 0)]
    print('df_filtered shape: ', df_filtered.shape)

    df_filtered = df_filtered.drop('Support Devices', axis=1)
    print(df_filtered.head())

    new_path = '/'.join(mimic_train_path.split('/')[:-1]) + '/mimic_train_version_one_five.csv'
    df_filtered.to_csv(new_path, index=False)

    df = pd.read_csv(mimic_test_path)
    print('df original shape: ', df.shape)    
    
    df_filtered = df[(df['BMI (kg/m2)'] != 0) & (df['Blood Pressure Max'] != 0) & (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] != 0) & (df['Weight (Lbs)'] != 0)]
    print('df_filtered shape: ', df_filtered.shape)

    normal_df_num = ((df['No Finding'] == 1) & (df['BMI (kg/m2)'] != 0) & (df['Blood Pressure Max'] != 0) & (df['Blood Pressure Min'] != 0) & (df['Height (Inches)'] != 0) & (df['Weight (Lbs)'] != 0)).sum()
    print('normal sample num', normal_df_num)

    df_filtered = df_filtered.drop('Support Devices', axis=1)
    print(df_filtered.head())

    new_path = '/'.join(mimic_test_path.split('/')[:-1]) + '/mimic_test_version_one_five.csv'
    df_filtered.to_csv(new_path, index=False)


def main():
    mimic_train_path = '/data3/hkim/datasets/preprocessed_mimic/mimic_third_train.csv'
    mimic_test_path = '/data3/hkim/datasets/preprocessed_mimic/mimic_third_test.csv'



    analysis_mimic(mimic_train_path, mimic_test_path)

    # make_mimic_1_5_dataset(mimic_train_path, mimic_test_path)


if __name__ == "__main__":
    main()