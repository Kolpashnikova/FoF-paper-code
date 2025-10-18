import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from avro.datafile import DataFileReader
from avro.io import DatumReader
import re

def synchronize_timestamps(sensor_df, offset_ms=0):
    sensor_df['timestamp'] = sensor_df['timestamp'] + pd.to_timedelta(offset_ms, unit='ms')
    return sensor_df

def load_empatica_data(avro_file):
    with open(avro_file, 'rb') as f:
        avro_reader = DataFileReader(f, DatumReader())
        records = list(avro_reader)
        avro_reader.close()

    rawData = records[0]['rawData']

    def convert_signal_to_df(signal_dict, col_name):
        values = signal_dict['values']
        start = signal_dict['timestampStart']
        freq = signal_dict['samplingFrequency']
        ts_ms = [start + i * 1_000_000 / freq for i in range(len(values))]
        ts_ms = pd.Series(ts_ms) // 1000
        df = pd.DataFrame({col_name: values})
        df['timestamp'] = pd.to_datetime(ts_ms, unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df

    acc_df = pd.DataFrame({
        'acc_x': rawData['accelerometer']['x'],
        'acc_y': rawData['accelerometer']['y'],
        'acc_z': rawData['accelerometer']['z']
    })
    acc_ts_ms = [rawData['accelerometer']['timestampStart'] + i*1_000_000/rawData['accelerometer']['samplingFrequency'] 
                 for i in range(len(acc_df))]
    acc_ts_ms = pd.Series(acc_ts_ms) // 1000
    acc_df['timestamp'] = pd.to_datetime(acc_ts_ms, unit='ms', utc=True)
    acc_df.set_index('timestamp', inplace=True)

    gyro_df = pd.DataFrame({
        'gyro_x': rawData['gyroscope']['x'],
        'gyro_y': rawData['gyroscope']['y'],
        'gyro_z': rawData['gyroscope']['z']
    })
    gyro_ts_ms = [rawData['gyroscope']['timestampStart'] + i*1_000_000/rawData['gyroscope']['samplingFrequency'] 
                  for i in range(len(gyro_df))]
    gyro_ts_ms = pd.Series(gyro_ts_ms) // 1000
    gyro_df['timestamp'] = pd.to_datetime(gyro_ts_ms, unit='ms', utc=True)
    gyro_df.set_index('timestamp', inplace=True)

    eda_df = convert_signal_to_df(rawData['eda'], 'eda_values')
    temp_df = convert_signal_to_df(rawData['temperature'], 'temperature_values')
    steps_df = convert_signal_to_df(rawData['steps'], 'steps_values')
    bvp_df = convert_signal_to_df(rawData['bvp'], 'bvp_values')

    sp = rawData['systolicPeaks']
    sp_time = pd.Series(sp['peaksTimeNanos'])
    sp_time_ms = sp_time // 1_000_000
    systolic_df = pd.DataFrame({'timestamp': pd.to_datetime(sp_time_ms, unit='ms', utc=True)})
    systolic_df['HR'] = 60_000_000_000 / systolic_df['timestamp'].diff().dt.total_seconds() / 1e9
    systolic_df.set_index('timestamp', inplace=True)

    all_dfs = [acc_df, gyro_df, eda_df, temp_df, steps_df, bvp_df, systolic_df]
    empatica_df = pd.concat(all_dfs, axis=1)
    return empatica_df

def load_pupil_data(data_dir, suffixes=None):
    if suffixes is None:
        suffixes = ['2', '3', '4', '5', '6', '']

    required_gaze_cols = ['fixation id', 'blink id', 'timestamp [ns]', 'confidence', 'norm_pos_x', 'norm_pos_y']
    required_fix_cols = ['fixation id', 'start fixation [ns]', 'end fixation [ns]', 
                         'duration fixation [ms]', 'fixation x [px]', 'fixation y [px]', 
                         'azimuth fixation [deg]', 'elevation fixation [deg]']
    required_blink_cols = ['blink id', 'start blink [ns]', 'end blink [ns]', 'duration blink [ms]']

    merged_data = {}

    participant_id = os.path.basename(os.path.normpath(data_dir))

    for suffix in suffixes:
        if suffix:
            gaze_file = f"gaze {suffix}.csv"
            fix_file = f"fixations {suffix}.csv"
            blink_file = f"blinks {suffix}.csv"
            merged_name = f"{participant_id}_merged_{suffix}"
        else:
            gaze_file = "gaze.csv"
            fix_file = "fixations.csv"
            blink_file = "blinks.csv"
            merged_name = f"{participant_id}_merged_base"
        
        gaze_path = os.path.join(data_dir, gaze_file)
        fix_path = os.path.join(data_dir, fix_file)
        blink_path = os.path.join(data_dir, blink_file)
        
        if not all(os.path.exists(p) for p in [gaze_path, fix_path, blink_path]):
            continue

        df_gaze = pd.read_csv(gaze_path)
        df_fix = pd.read_csv(fix_path)
        df_blinks = pd.read_csv(blink_path)

        if df_fix.shape[1] == 8:
            df_fix.columns = [
                'fixation id', 'start fixation [ns]', 'end fixation [ns]',
                'duration fixation [ms]', 'fixation x [px]', 'fixation y [px]',
                'azimuth fixation [deg]', 'elevation fixation [deg]'
            ]
        else:
            raise ValueError(f"[ERROR] Fixation file '{fix_file}' does not have the expected number of columns.")

        if df_blinks.shape[1] == 4:
            df_blinks.columns = [
                'blink id', 'start blink [ns]', 'end blink [ns]',
                'duration blink [ms]'
            ]
        else:
            raise ValueError(f"[ERROR] Blink file '{blink_file}' does not have the expected number of columns.")

        missing_fix_cols = set(required_fix_cols) - set(df_fix.columns)
        missing_blink_cols = set(required_blink_cols) - set(df_blinks.columns)
        if missing_fix_cols:
            raise ValueError(f"[ERROR] Missing columns in fixations file: {missing_fix_cols}")
        if missing_blink_cols:
            raise ValueError(f"[ERROR] Missing columns in blink file: {missing_blink_cols}")

        merged_gaze = pd.merge(df_gaze, df_fix, on='fixation id', how='left')
        merged_gaze = pd.merge(merged_gaze, df_blinks, on='blink id', how='left')

        if 'confidence' in merged_gaze.columns:
            merged_gaze = merged_gaze[merged_gaze['confidence'] > 0.8].copy()

        merged_gaze['participant_full_id'] = participant_id
        merged_gaze['file_id'] = suffix if suffix else 'base'

        if 'timestamp [ns]' in merged_gaze.columns:
            merged_gaze.sort_values('timestamp [ns]', inplace=True)
        else:
            print(f"[WARNING] 'timestamp [ns]' column not found, consider verifying time alignment.")

        merged_data[merged_name] = merged_gaze

    return merged_data

def load_all_data(base_dir, folders, suffixes):
    combined_df = {}

    for folder in folders:
        data_dir = os.path.join(base_dir, folder)

        pupil_data_dict = load_pupil_data(data_dir, suffixes=suffixes)
        if not pupil_data_dict:
            continue

        avro_files = glob.glob(os.path.join(data_dir, "raw", "1-1-[0-9][0-9]_*.avro"))
        if not avro_files:
            continue

        empatica_dfs = []
        for af in avro_files:
            try:
                edf = load_empatica_data(af)
                empatica_dfs.append(edf)
            except Exception as e:
                print(f"[ERROR] Failed to load Empatica data from {af}: {e}")
        
        if not empatica_dfs:
            continue

        empatica_df = pd.concat(empatica_dfs).sort_index()

        for name, gaze_df in pupil_data_dict.items():
            if 'timestamp [ns]' in gaze_df.columns:
                gaze_df['timestamp_ms'] = gaze_df['timestamp [ns]'] // 1_000_000
                gaze_df['datetime_ms'] = pd.to_datetime(gaze_df['timestamp_ms'], unit='ms', utc=True)
                gaze_df.set_index('datetime_ms', inplace=True)
            else:
                continue

            start_time = max(empatica_df.index.min(), gaze_df.index.min())
            end_time = min(empatica_df.index.max(), gaze_df.index.max())

            if start_time >= end_time:
                continue

            common_freq = '10ms'

            numeric_cols = gaze_df.select_dtypes(include='number').columns
            non_numeric_cols = gaze_df.columns.difference(numeric_cols)
            
            gaze_numeric = gaze_df[numeric_cols]
            gaze_non_numeric = gaze_df[non_numeric_cols]
            
            gaze_numeric_sliced = gaze_numeric[start_time:end_time]
            empatica_sliced = empatica_df[start_time:end_time]
            
            gaze_res = (
                gaze_numeric_sliced
                .resample(common_freq)
                .mean(numeric_only=True)
                .interpolate(method='linear')
            )
            
            empatica_res = (
                empatica_sliced
                .resample(common_freq)
                .mean(numeric_only=True)
                .interpolate(method='linear')
            )
            
            participant_id = gaze_df['participant_full_id'].iloc[0] if 'participant_full_id' in gaze_df else folder
            file_id = gaze_df['file_id'].iloc[0] if 'file_id' in gaze_df else name.split('_')[-1]
            
            final_df = pd.concat([gaze_res, empatica_res], axis=1)
            
            final_df['participant_full_id'] = participant_id
            final_df['file_id'] = file_id

            participant_full_id = (
                gaze_df['participant_full_id'].iloc[0]
                if 'participant_full_id' in gaze_df.columns
                else folder
            )
            
            file_id = (
                gaze_df['file_id'].iloc[0]
                if 'file_id' in gaze_df.columns
                else name.split('_')[-1]
            )
            
            final_df['participant_full_id'] = participant_full_id
            final_df['file_id'] = file_id
            final_df['elapsed_time_ms'] = (final_df.index - final_df.index[0]) / pd.Timedelta('1ms')
            
            combined_key = f"{participant_full_id}_{file_id}"
            combined_df[combined_key] = final_df

    return combined_df

if __name__ == "__main__":
    base_dir = "data"
    folders = [
        "493_1_1_03",
        "493_1_1_10",
        "493_1_1_11",
        "493_1_1_15",
        "493_1_1_16",
        "493_1_1_17"
    ]
    suffixes = ['2', '3', '4', '5', '6', '']  

    full_datasets = load_all_data(base_dir, folders, suffixes)

all_FoF = pd.concat(full_datasets.values(), ignore_index=True)

demographics = pd.read_excel('data/demographics_data.xlsx')
demographics['participant_full_id'] = demographics['participant_full_id_y']

df_m = pd.merge(
    all_FoF,
    demographics,
    on='participant_full_id',
    how='left'
)

fof_columns = [f'fof{i}' for i in range(1, 17)]
df_m['FES_sum'] = df_m[fof_columns].sum(axis=1)
df_m['FoF_Category'] = np.where(df_m['FES_sum'] < 40, 'Low FoF', 'High FoF')
df_m['FoF_Cat'] = np.where(df_m['FES_sum'] < 40, 0, 1)

df_m.to_parquet("data/full_data_combined.parquet")

df_m = pd.read_parquet("data/full_data_combined.parquet")
df_m['elapsed_time'] = pd.to_timedelta(df_m['elapsed_time_ms'], unit='ms')

df = df_m[~df_m['participant_full_id'].isin(['493_1_1_12_01', '493_1_1_12_02', '493_1_1_14'])]
df = df[~((df['participant_full_id'] == '493_1_1_11') & (df['file_id'] == 'base'))]
df = df[~((df['participant_full_id'] == '493_1_1_11') & (df['file_id'] == '2'))]

def clean_column_name(col):
    col = re.sub(r'\[.*?\]', '', col)
    col = col.strip()
    col = col.replace(' ', '_')
    return col

df = df.rename(columns={c: clean_column_name(c) for c in df.columns})

df_clean = df[['acc_x', 'acc_y', 'acc_z',
    'eda_values', 'temperature_values', 'steps_values',
    'azimuth', 'elevation', 'bvp_values', 'HR',
    'duration_fixation', 'duration_blink', 'participant_full_id', 'file_id', 'elapsed_time', 'FoF_Cat']]

df_clean[['duration_fixation', 'duration_blink']] = df_clean[['duration_fixation', 'duration_blink']].fillna(0)

df_clean = df_clean[df_clean.notna().all(axis=1)]

def iir_lowpass_filter(signal, cutoff_freq, fs, order=7):
    nyq = fs / 2.0
    normalized_cutoff = cutoff_freq / nyq
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

fs = 100
cutoff_freq = 5

df_clean = df_clean.assign(**{f'filtered_{i}': np.nan for i in [
    'elevation', 'bvp_values', 'HR', 'acc_x', 'acc_y', 'acc_z', 
    'eda_values', 'azimuth'
]})

cols_to_filter = ['elevation', 'bvp_values', 'HR', 'acc_x', 'acc_y', 'acc_z', 'eda_values', 'azimuth']

for i in cols_to_filter:
    for part in df_clean['participant_full_id'].unique():
        for file in df_clean.loc[df_clean['participant_full_id'] == part, 'file_id'].unique():
            mask = (df_clean['participant_full_id'] == part) & (df_clean['file_id'] == file)
            signal = df_clean.loc[mask, i].interpolate().to_numpy()
            filtered_signal = iir_lowpass_filter(signal, cutoff_freq, fs, order=7)
            df_clean.loc[mask, f'filtered_{i}'] = filtered_signal

df_clean['filtered_temperature_values'] = df_clean['temperature_values']
df_clean['filtered_steps_values'] = df_clean['steps_values']
df_clean['filtered_duration_fixation'] = df_clean['duration_fixation']
df_clean['filtered_duration_blink'] = df_clean['duration_blink']

def tercile_cuts(x):
    q1 = x.quantile(1/3)
    q2 = x.quantile(2/3)
    
    if q1 == q2:
        return pd.Series([2]*len(x), index=x.index)
    else:
        return pd.cut(x, bins=[-np.inf, q1, q2, np.inf], labels=[1,2,3])

variables_to_recode = ['filtered_elevation', 'filtered_HR']

for var in variables_to_recode:
    new_col = f"{var}_cat"
    grouped = df_clean.groupby('participant_full_id')[var]
    df_clean[new_col] = grouped.transform(tercile_cuts)

df_clean['Potential_FoF'] = np.where(
    (df_clean['filtered_elevation_cat'] == 1) & 
    (df_clean['filtered_HR_cat'] == 3),
    1, 
    0
)

df_clean.to_parquet("data/model_data_final_full.parquet")

cols_to_keep = [
    'filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z', 
    'filtered_eda_values', 'filtered_temperature_values', 'filtered_steps_values', 
    'filtered_azimuth', 'filtered_duration_fixation', 'filtered_duration_blink', 
    'Potential_FoF'
]

df_clean = df_clean[cols_to_keep]
df_clean = df_clean.dropna()
df_clean.to_parquet("data/model_data_final.parquet")