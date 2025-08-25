import os
import re
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.datasets import dump_svmlight_file

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_df(file_path):
    with open(file_path) as f:
        samples = []
        for line in f.readlines():
            if line.startswith('@inputs'):
                columns = re.split(',\s?', line.split('@inputs ')[-1].strip())
            if line.startswith('@output'):
                outputs = line.split()[-1].strip()
                columns.append(outputs)

            if not line.startswith('@'):
                if '<null>' in line:
                        continue
                samples.append(line.strip().split(','))
        
    #  df_samples = pd.DataFrame(samples, columns=columns, dtype=float)
    df_samples = pd.DataFrame(samples, columns=columns)
    for col in columns:
        df_samples[col] = pd.to_numeric(df_samples[col], errors='ignore')
    return outputs, df_samples


def get_train_test_df(dataset_name, fold_idx, source_folder='raw_data'):
    
    train_file_path = os.path.join(f'{source_folder}/{dataset_name}-5-fold', f'{dataset_name}-5-{fold_idx}tra.dat')
    test_file_path = os.path.join(f'{source_folder}/{dataset_name}-5-fold', f'{dataset_name}-5-{fold_idx}tst.dat')
    
    outputs, train_df = get_df(train_file_path)
    outputs, test_df = get_df(test_file_path)

    return outputs, train_df, test_df


def get_df_stats(df, output_col, dataset_name):
    n_samples = df.shape[0]
    n_numeric = 0
    n_nominal = 0
    numeric = []
    nomial = []
    for col in df.columns:
        if col == output_col:
            label_stats = df[col].value_counts()
            num_class = len(label_stats)
            label_dist = '/'.join(label_stats.values.astype(str).tolist())
        
        elif dataset_name in ['balance', 'ecoli', 'hayes-roth']: 
            numeric.append(col)
            n_numeric += 1         

        elif dataset_name in ['lymphography']:
            if isinstance(df[col][0], str):
                nomial.append(col)
                n_nominal += 1
                 
            else:
                numeric.append(col)
                n_numeric += 1
        else: 
            if  (df[col].nunique() <= 8) or (isinstance(df[col][0], str)): 
                nomial.append(col)
                n_nominal += 1
            else:
                numeric.append(col)
                n_numeric += 1
        n_attr = n_numeric + n_nominal

    print(f'#EX.: {n_samples}, #Atts: {n_attr}, #Num: {n_numeric}, #Nom: {n_nominal}, #CI: {num_class}, #Dc.: {label_dist}')
    return nomial, numeric

def process_keel(dataset_name, source_folder='raw_data', save_folder='svm_data'):

    if save_folder == 'svm_data':
        nomial_handle = 'one-hot' 
    else:
        nomial_handle = 'ordinal' 

    save_dir = os.path.join(save_folder, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    for fold_idx in range(1, 6):
        outputs, train_df, test_df = get_train_test_df(dataset_name, fold_idx, source_folder)

        if fold_idx == 1:
            full_df = pd.concat([train_df, test_df], axis=0)
            full_df.reset_index(drop=True, inplace=True)
            nomial, numeric = get_df_stats(full_df, output_col=outputs, dataset_name=dataset_name)
            
            le = LabelEncoder()
            le.fit(full_df[outputs])

            if nomial:
                if nomial_handle == 'one-hot':
                    onehot = OneHotEncoder(drop='if_binary')
                    onehot.fit(full_df[nomial])
                else:
                    ordenc = OrdinalEncoder()
                    ordenc.fit(full_df[nomial])

        train_df[outputs] = le.transform(train_df[outputs]) + 1
        test_df[outputs] = le.transform(test_df[outputs]) + 1

        if numeric:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df[numeric] = scaler.fit_transform(train_df[numeric])
            test_df[numeric] = scaler.transform(test_df[numeric])

        if nomial:
            if nomial_handle == 'one-hot':
                # convert encoded array into dataframe
                train_encoded = pd.DataFrame(onehot.transform(train_df[nomial]).toarray(), columns=onehot.get_feature_names_out(nomial))
                test_encoded = pd.DataFrame(onehot.transform(test_df[nomial]).toarray(), columns=onehot.get_feature_names_out(nomial))

                # drop original one hot columns 
                train_df = train_df.drop(columns=nomial).reset_index(drop=True)
                test_df = test_df.drop(columns=nomial).reset_index(drop=True)

                # concat dataframe
                train_df = pd.concat([train_df, train_encoded], axis=1)
                test_df = pd.concat([test_df, test_encoded], axis=1)
            else:
                train_df[nomial] = ordenc.transform(train_df[nomial])
                test_df[nomial] = ordenc.transform(test_df[nomial])

        cols = train_df.columns
        inputs = cols[cols != outputs]

        trainX, trainY = train_df[inputs], train_df[outputs]
        testX, testY = test_df[inputs], test_df[outputs]

        dump_svmlight_file(trainX, trainY, os.path.join(save_dir, f'train_{dataset_name}_{fold_idx}.svm'), zero_based=False)
        dump_svmlight_file(testX, testY , os.path.join(save_dir, f'test_{dataset_name}_{fold_idx}.svm'), zero_based=False)
    print('Preprocessing finished.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets and save them in SVM format.")
    parser.add_argument("--datasets", nargs='+', required=True, help="List of dataset names.")
    parser.add_argument("--source_folder", type=str, required=True, help="Directory of raw data.")
    parser.add_argument("--save_folder", type=str, required=True, help="Directory to save processed SVM files.")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        print('Dataset name:', dataset_name)
        process_keel(dataset_name, source_folder=args.source_folder, save_folder=args.save_folder)