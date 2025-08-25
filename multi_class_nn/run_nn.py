import os
import json
import torch
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from data_utils import SVMFormatDataset
from trainer import train_multiclass
from evaluation import predict_multiclass

def run_nn(datasets, method='ovo', source_folder='svm_data', dest_folder='.', stop_criteria='Accuracy'):

    for dataset_name in tqdm(datasets):

        save_dir = f'{dest_folder}/svm-predict-NN-{stop_criteria}/{dataset_name}/{method}'
        model_dir = f'models/{dataset_name}/{method}'
        os.system(f'rm -r {save_dir}')
        os.makedirs(save_dir, exist_ok=True)
    
        for i in range(1, 6):

            train_file = f'{source_folder}/{dataset_name}/train_{dataset_name}_{i}.svm'
            test_file = f'{source_folder}/{dataset_name}/test_{dataset_name}_{i}.svm'
        
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            train_dataset = SVMFormatDataset(train_file)
            test_dataset = SVMFormatDataset(test_file, num_features=train_dataset.get_num_features())

            num_classes = len(torch.unique(train_dataset.labels))

            batch_size = 8
            epochs = 40
            patience = 10
            learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

            print(f"Training method: {method}")
            model_paths, config = train_multiclass(
                train_dataset, 
                num_classes, 
                method=method, 
                save_dir=model_dir, 
                device=device, 
                batch_size=batch_size, 
                epochs=epochs, 
                patience=patience,
                stop_criteria=stop_criteria,
                learning_rates=learning_rates
            )

            print(f"Models saved to: {model_dir}")

            # Prediction
            print("Starting prediction...")
            target = test_dataset.labels.numpy()
            predictions, decision_values = predict_multiclass(
                                                test_dataset, 
                                                num_classes, 
                                                model_paths, 
                                                method=method, 
                                                device=device
                                            )
            
            macrof1 = f1_score(target, predictions, average='macro')
            accuracy = sum(p == l for p, l in zip(target, predictions)) / len(target)
            print("Accuracy:", accuracy)
            print("Macro-F1:", macrof1)

            label_info = {
                'target': (target + 1).tolist(), 
                'predict': (predictions + 1).tolist(),
                'decision_val': decision_values,
                'config': [config]
            }

            file_name = os.path.basename(test_file).split('.')[0]
            label_file = os.path.join(save_dir, f'{file_name} label.json')

            with open(label_file, 'w') as f:
                json.dump(label_info, f, indent=4)
        os.system(f'rm -r {model_dir}')

if __name__ == "__main__":
    source_folder='/home/ktdev/ovo-ovr/svm_data'
    dest_folder = '/home/ktdev/ovo-ovr/multi_class_nn'
    dest_folder = '/home/ktdev/ovo-ovr'
    datasets = ['autos', 'balance', 'car', 'cleveland', 'dermatology', 'dna', 'ecoli', 'flare', 'glass', 'hayes-roth', 'hcv', 'letter', 'lymphography', 'newthyroid', 'segment', 'shuttle', 'thyroid', 'zoo']
    # exclude letter due to larger size, dna to larger features
    datasets = ['autos', 'balance', 'car', 'cleveland', 'dermatology', 'ecoli', 'flare', 'glass', 'hayes-roth', 'hcv', 'lymphography', 'newthyroid', 'segment', 'shuttle', 'thyroid', 'zoo', "vehicle"]#, "satimage"]
    # methods = ['naive-ovo-v3']#['ovo', 'ovr', 'naive-ovr', 'naive-ovo', 'naive-ovo-v2', 'naive-ovo-v3']

    methods = ['naive-ovo-v2']#['naive-ovr', 'naive-ovo-v3', 'naive-ovr-v2']
    for monitor in ['Accuracy', 'MacroF1']:
        for method in methods: 
            run_nn(datasets, method=method, source_folder=source_folder, dest_folder=dest_folder, stop_criteria=monitor)
