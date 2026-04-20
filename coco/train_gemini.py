import argparse
import os
import sys
import numpy as np # Hinzugefügt für Geometric Mean

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

def build_dataloader(json_path, batch_size, shuffle=True, num_workers=1, use_logic_augmented_features=False, use_rule_components=False, use_clip_features=False, debug=False):
    dataset = CocoLogicDataset(
        json_path=json_path,
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


def compute_class_weights(dataloader, device):
    """Berechnet die pos_weights für BCEWithLogitsLoss basierend auf dem Trainingsset."""
    print("Computing class weights from training set...")
    all_labels = []
    for _, labels in dataloader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0) # Shape: (Total_Samples, Num_Classes)
    
    num_positives = all_labels.sum(dim=0)
    num_negatives = all_labels.shape[0] - num_positives
    
    # pos_weight = negative / positive. Clamp auf min=1.0 um Division durch 0 zu verhindern.
    pos_weight = num_negatives / torch.clamp(num_positives, min=1.0)
    print(f"Class positive weights computed: {pos_weight.cpu().numpy().round(2)}")
    return pos_weight.to(device)


def train(
    train_json, val_json, batch_size=128, lr=1e-3, epochs=10, device=None, model_type="linear",
    hidden_dims=(256, 128), dropout=0.0, use_logic_augmented_features=False,
    use_rule_components=False, use_clip_features=False, debug=False,
    patience=10 # Early Stopping Parameter
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(
        train_json, batch_size=batch_size, shuffle=True,
        use_logic_augmented_features=use_logic_augmented_features,
        use_rule_components=use_rule_components, use_clip_features=use_clip_features, debug=debug,
    )
    val_loader = build_dataloader(
        val_json, batch_size=batch_size, shuffle=False,
        use_logic_augmented_features=use_logic_augmented_features,
        use_rule_components=use_rule_components, use_clip_features=use_clip_features, debug=debug,
    )

    sample_item = next(iter(train_loader))
    input_dim = sample_item[0].shape[1]
    output_dim = sample_item[1].shape[1]

    model = make_model(
        model_type=model_type, input_dim=input_dim, output_dim=output_dim,
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    
    # 1. Class Weights berechnen und in die Loss-Funktion einbauen
    pos_weights = compute_class_weights(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_geom_mean = 0.0
    patience_counter = 0

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
        
        all_preds = []
        all_labels = []

        counter = 0
        
        with torch.no_grad():
            for categories, labels in val_loader:
                categories = categories.to(device)
                labels = labels.to(device)
                outputs = model(categories)
                # if counter < 2:
                #     print(outputs.mean().item())
                #     counter += 1
                loss = criterion(outputs, labels)
                val_loss += loss.item() * categories.size(0)
                
                # Sammeln für Balanced Accuracy
                predictions = (outputs >= 0.0).float() # Äquivalent zu sigmoid(outputs) >= 0.5 # > 0.5
                all_preds.append(predictions)
                all_labels.append(labels)

        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # 2. Balanced Accuracy pro Klasse berechnen
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        balanced_accs = []
        for c in range(output_dim):
            preds_c = all_preds[:, c]
            labels_c = all_labels[:, c]
            
            tp = ((preds_c == 1) & (labels_c == 1)).sum().float()
            tn = ((preds_c == 0) & (labels_c == 0)).sum().float()
            fp = ((preds_c == 1) & (labels_c == 0)).sum().float()
            fn = ((preds_c == 0) & (labels_c == 1)).sum().float()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0)
            
            b_acc = (sensitivity + specificity) / 2.0
            balanced_accs.append(b_acc.item())
            
        # 3. Geometric Mean über alle Balanced Accuracies
        b_accs_arr = np.array(balanced_accs)
        geom_mean = np.exp(np.mean(np.log(np.clip(b_accs_arr, 1e-8, 1.0)))) * 100.0

        print(
            f"Epoch {epoch:03d}/{epochs:03d}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={avg_val_loss:.4f}, Geom. Mean Bal. Acc={geom_mean:.2f}%"
        )
        
        # 4. Early Stopping Logik
        if geom_mean > best_geom_mean:
            best_geom_mean = geom_mean
            patience_counter = 0
            # Optional: Hier torch.save(model.state_dict(), 'best_model.pth') einbauen
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered nach Epoche {epoch}! Bestes Geom. Mean: {best_geom_mean:.2f}%")
                break

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a COCOLogic predictor from categories to rule labels.")
    parser.add_argument("--train_json", type=str, default="coco/cocologic_train_final.json")
    parser.add_argument("--val_json", type=str, default="coco/cocologic_test_final.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--model_type", type=str, choices=["linear", "mlp"], default="linear",
        help="Choose model architecture: linear or mlp.",
    )
    parser.add_argument(
        "--hidden_dims", type=str, default="128",
        help="Comma-separated hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout probability for MLP hidden layers.",
    )
    parser.add_argument(
        "--use_logic_augmented_features", action="store_true",
        help="Augment input features with logic-derived features.",
    )
    parser.add_argument(
        "--use_rule_components", action="store_true",
        help="Include rule_components from JSON.",
    )
    parser.add_argument(
        "--use_clip_features", action="store_true",
        help="Use precomputed CLIP similarity features as input.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print debug information.",
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

    train(
        train_json=train_json, val_json=val_json, batch_size=args.batch_size,
        lr=args.lr, epochs=args.epochs, device=args.device,
        model_type=args.model_type, hidden_dims=hidden_dims,
        dropout=args.dropout, use_logic_augmented_features=args.use_logic_augmented_features,
        use_rule_components=args.use_rule_components, use_clip_features=args.use_clip_features,
        debug=args.debug,
    )