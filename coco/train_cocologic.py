import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cocologic_dataset import CocoLogicDataset, collate_cocologic


class RulePredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MLPRulePredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_pos_weights(train_loader, num_classes=10):
    pos_counts = torch.zeros(num_classes)
    total_samples = 0
    for _, labels in train_loader:
        pos_counts += labels.sum(dim=0)
        total_samples += labels.size(0)
    neg_counts = total_samples - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-6)  # avoid div by zero
    return pos_weights


def compute_balanced_geom_mean_acc(outputs, labels, num_classes=10):
    predictions = torch.sigmoid(outputs) >= 0.5
    bal_accs = []
    for cls in range(num_classes):
        pred_cls = predictions[:, cls]
        true_cls = labels[:, cls]
        tp = ((pred_cls == 1) & (true_cls == 1)).sum().float()
        tn = ((pred_cls == 0) & (true_cls == 0)).sum().float()
        fp = ((pred_cls == 1) & (true_cls == 0)).sum().float()
        fn = ((pred_cls == 0) & (true_cls == 1)).sum().float()
        sens = tp / (tp + fn + 1e-6)  # sensitivity
        spec = tn / (tn + fp + 1e-6)  # specificity
        bal_acc = (sens + spec) / 2
        bal_accs.append(bal_acc.item())
    # Geometric mean
    geom_mean = np.prod(bal_accs) ** (1 / len(bal_accs))
    return geom_mean


def build_dataloader(json_path, batch_size, shuffle=True, num_workers=1, use_logic_augmented_features=False, use_rule_components=False, use_clip_features=False, debug=False):
    dataset = CocoLogicDataset(
        json_path=json_path,
        use_image=False,
        use_logic_augmented_features=use_logic_augmented_features,
        use_rule_components=use_rule_components,
        use_clip_features=use_clip_features,
        debug=debug,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_cocologic,
    )


def make_model(model_type: str, input_dim: int, output_dim: int, hidden_dims=None, dropout: float = 0.0):
    model_type = model_type.lower()
    if model_type == "linear":
        return RulePredictor(input_dim=input_dim, output_dim=output_dim)
    if model_type == "mlp":
        return MLPRulePredictor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims or (128,),
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_type: {model_type}. Choose 'linear' or 'mlp'.")


def train(
    train_json,
    val_json,
    batch_size=128,
    lr=1e-3,
    epochs=10,
    device=None,
    model_type="linear",
    hidden_dims=(256, 128),
    dropout=0.0,
    use_logic_augmented_features=False,
    use_rule_components=False,    use_clip_features=False,    debug=False,    patience=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(
        train_json,
        batch_size=batch_size,
        shuffle=True,
        use_logic_augmented_features=use_logic_augmented_features,
        use_rule_components=use_rule_components,
        use_clip_features=use_clip_features,
        debug=debug,
    )
    val_loader = build_dataloader(
        val_json,
        batch_size=batch_size,
        shuffle=False,
        use_logic_augmented_features=use_logic_augmented_features,
        use_rule_components=use_rule_components,
        use_clip_features=use_clip_features,
        debug=debug,
    )

    sample_item = next(iter(train_loader))
    input_dim = sample_item[0].shape[1]
    output_dim = sample_item[1].shape[1]

    # Compute positive weights for unbalanced classes
    pos_weights = compute_pos_weights(train_loader, num_classes=output_dim).to(device)
    print(f"Positive weights: {pos_weights}")

    model = make_model(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for categories, labels in train_loader:
            categories = categories.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(categories)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * categories.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for categories, labels in val_loader:
                categories = categories.to(device)
                labels = labels.to(device)
                outputs = model(categories)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * categories.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_val_loss = val_loss / len(val_loader.dataset)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_bal_geom_acc = compute_balanced_geom_mean_acc(all_outputs, all_labels, num_classes=output_dim)

        print(
            f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, balanced_geom_mean_acc={val_bal_geom_acc:.4f}"
        )

        if patience is not None:
            if val_bal_geom_acc > best_acc:
                best_acc = val_bal_geom_acc
                counter = 0
            else:
                counter += 1
            if counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model, val_bal_geom_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a COCOLogic predictor from categories to rule labels.")
    parser.add_argument("--train_json", type=str, default="coco/cocologic_train_final.json")
    parser.add_argument("--val_json", type=str, default="coco/cocologic_test_final.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--lr_list",
        type=str,
        default=None,
        help="Comma-separated list of learning rates to try, e.g., '0.001,0.0001,0.01'. If provided, trains with each LR and reports best.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["linear", "mlp"],
        default="linear",
        help="Choose model architecture: linear or mlp.",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="128",
        help="Comma-separated hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability for MLP hidden layers.",
    )
    parser.add_argument(
        "--use_logic_augmented_features",
        action="store_true",
        help="Augment input features with logic-derived features to help linear models handle non-linear rule conditions.",
    )
    parser.add_argument(
        "--use_rule_components",
        action="store_true",
        help="Include rule_components from JSON as additional input features (binary indicators of satisfied positive components).",
    )
    parser.add_argument(
        "--use_clip_features",
        action="store_true",
        help="Use precomputed CLIP similarity features as input instead of category counts.",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for the first few samples showing computed logic features vs true labels.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )
    args = parser.parse_args()

    train_json = os.path.abspath(args.train_json)
    val_json = os.path.abspath(args.val_json)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())

    print("Using train JSON:", train_json)
    print("Using val JSON:", val_json)
    print("Using device:", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Using model type:", args.model_type)
    if args.model_type == "mlp":
        print("Using hidden dims:", hidden_dims, "dropout:", args.dropout)
    print("Using logic augmented features:", args.use_logic_augmented_features)
    print("Using rule components:", args.use_rule_components)
    print("Using CLIP features:", args.use_clip_features)

    if args.lr_list:
        lrs = [float(x.strip()) for x in args.lr_list.split(',')]
        best_acc = 0
        best_lr = None
        for lr in lrs:
            print(f"\nTrying LR: {lr}")
            model, acc = train(
                train_json=train_json,
                val_json=val_json,
                batch_size=args.batch_size,
                lr=lr,
                epochs=args.epochs,
                device=args.device,
                model_type=args.model_type,
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                use_logic_augmented_features=args.use_logic_augmented_features,
                use_rule_components=args.use_rule_components,
                use_clip_features=args.use_clip_features,
                debug=args.debug,
                patience=args.patience,
            )
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        print(f"\nBest LR: {best_lr} with balanced_geom_mean_acc: {best_acc:.4f}")
    else:
        model, acc = train(
            train_json=train_json,
            val_json=val_json,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            device=args.device,
            model_type=args.model_type,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            use_logic_augmented_features=args.use_logic_augmented_features,
            use_rule_components=args.use_rule_components,
            use_clip_features=args.use_clip_features,
            debug=args.debug,
            patience=args.patience,
        )
