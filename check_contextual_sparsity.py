import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append('./multi_class_nn')
from data_utils import SVMFormatDataset, stratified_split
from model import FeedforwardNN
from trainer import train_model


def read_dataset(source_folder: str, dataset_name: str, fold: int = 1):
    """Build train/test datasets from SVM files for the given dataset and fold."""
    train_file = f'{source_folder}/{dataset_name}/train_{dataset_name}_{fold}.svm'
    test_file  = f'{source_folder}/{dataset_name}/test_{dataset_name}_{fold}.svm'

    train_dataset = SVMFormatDataset(train_file)
    test_dataset  = SVMFormatDataset(test_file, num_features=train_dataset.get_num_features())
    return train_dataset, test_dataset

def do_train(
    train_dataset: SVMFormatDataset,
    num_classes: int,
    model_dir: str,
    stop_criteria: str,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> str:
    """
    Train an initial model with a validation split, then retrain on the full
    training set and save both checkpoints into `model_dir`.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Split into train/val
    train_subset, val_subset = stratified_split(train_dataset, test_size=0.2, seed=42)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size)

    # First-stage training with validation
    model_path = os.path.join(model_dir, "model.pth")

    best_metric = -1 * np.inf
    best_epoch = 0
    best_lr = 0
    for lr in learning_rate:
        model = FeedforwardNN(input_size=train_dataset.get_num_features(), output_size=num_classes)
        model, best_epoch_one, best_metric_one = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=lr,
            epochs=epochs,
            patience=patience,
            stop_criteria=stop_criteria,
            save_path=model_path,
        )
        if best_metric_one >= best_metric:
            best_metric = best_metric_one
            best_epoch = best_epoch_one
            best_lr = lr

    # Retrain on the full training set (no validation)
    print(f"retrain with learning_rate = {best_lr}, epoch = {best_epoch}")
    full_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    retrain_model_path = os.path.join(model_dir, "retrain_model.pth")

    retrain_model = FeedforwardNN(input_size=train_dataset.get_num_features(), output_size=num_classes)
    retrain_model, _, _ = train_model(
        model=retrain_model,
        train_loader=full_train_loader,
        val_loader=None,
        learning_rate=best_lr,
        epochs=best_epoch,  # reuse best epoch from stage-1
        patience=patience,
        stop_criteria=stop_criteria,
        save_path=retrain_model_path,
    )

    return 


def evaluate_topk_features(
    model_dir: str,
    test_dataset: SVMFormatDataset,
    topk_ratio: np.ndarray,
    num_classes: int,
    batch_size: int,
):
    """
    Load the retrained model from `model_dir`, evaluate on the test set across
    different feature densities by masking top-k activations, and return predictions.
    """
    # Load checkpoint to CPU first (portable), then move to the chosen device
    retrain_model_path = os.path.join(model_dir, "retrain_model.pth")
    model = FeedforwardNN(input_size=test_dataset.get_num_features(), output_size=num_classes)
    state_dict = torch.load(retrain_model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Register a forward hook to capture intermediate features (assumes model has `fc1`)
    features = []
    def forward_hook(module, data_input, data_output):
        features.append(data_output)
    assert hasattr(model, "fc1"), "Model must expose an attribute `fc1` for feature extraction."
    handle = model.fc1.register_forward_hook(forward_hook)

    # Ensure ratios are sorted and unique for consistent plotting
    topk_ratio = np.unique(np.sort(topk_ratio))

    all_preds = {float(k_ratio): [] for k_ratio in topk_ratio}
    full_preds, true_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            features.clear()
            inputs, labels = inputs.to(device), labels.to(device)

            # Baseline forward to populate `features`
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            full_preds.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            # Take activations captured by the hook
            feature = features[0]  # shape: [B, F]
            for k_ratio in topk_ratio:
                k = int(k_ratio * feature.shape[1])
                if k <= 0:
                    continue

                # Select top-k features per sample
                _, indices = torch.topk(feature, k=k, dim=1)
                mask = torch.zeros_like(feature)
                mask.scatter_(1, indices, 1).bool()

                masked_outputs = model(inputs, mask) 
                masked_preds = torch.argmax(masked_outputs, dim=1)
                all_preds[float(k_ratio)].extend(masked_preds.cpu().numpy())

    handle.remove()
    return all_preds, true_labels

def plot(
    all_preds: dict,
    true_labels: list,
    dataset_name: str,
    model_dir: str,
):
    """Compute accuracy vs. density and save a plot into `model_dir`."""
    # Keep ratios ordered for plotting
    ratios = sorted(all_preds.keys())
    accuracy_across_k = []
    for r in ratios:
        correct = sum(int(p == l) for p, l in zip(all_preds[r], true_labels))
        acc = 100.0 * correct / max(1, len(true_labels))
        accuracy_across_k.append(acc)

    # Plot and save
    plt.rcParams['font.size'] = 16
    plt.plot(ratios, accuracy_across_k, marker='o')
    plt.title(dataset_name)
    plt.xlabel('density')
    plt.ylabel('accuracy')
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, f'{dataset_name}.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    # Parameters
    source_folder = '/home/ktdev/optml-dejavu/svm_data'
    dataset_name  = 'aloi'          # options: 'segment', 'vehicle', 'aloi'
    stop_criteria = 'Accuracy'
    batch_size    = 32
    epochs        = 40
    patience      = 10
    lr_list      = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2] # try values like [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    topk_ratio    = np.arange(0.1, 1.0, 0.05, dtype=float)
    model_dir     = os.path.join("models", dataset_name)

    # Read datasets
    train_dataset, test_dataset = read_dataset(source_folder, dataset_name, fold=1)
    num_classes = int(len(torch.unique(train_dataset.labels)))

    # Train and save checkpoints
    do_train(
        train_dataset=train_dataset,
        num_classes=num_classes,
        model_dir=model_dir,
        stop_criteria=stop_criteria,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        learning_rate=lr_list,
    )

    # Evaluate using saved retrained model
    all_preds, true_labels = evaluate_topk_features(
        model_dir=model_dir,
        test_dataset=test_dataset,
        topk_ratio=topk_ratio,
        num_classes=num_classes,
        batch_size=batch_size,
    )

    # Plot results
    plot(
        all_preds=all_preds,
        true_labels=true_labels,
        dataset_name=dataset_name,
        model_dir=model_dir,
    )