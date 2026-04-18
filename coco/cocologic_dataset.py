import json
import os
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CocoLogicDataset(Dataset):
    """PyTorch dataset for COCOLogic JSON files.

    This dataset can return either:
      - category count vectors and rule label vectors
      - optionally image pixels with the same target labels
    """

    def __init__(
        self,
        json_path: str,
        image_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        use_image: bool = False,
        use_logic_augmented_features: bool = False,
        use_rule_components: bool = False,
        use_clip_features: bool = False,
        debug: bool = False,
    ):
        with open(json_path, "r") as f:
            data = json.load(f)

        if use_image and image_dir is None:
            raise ValueError("image_dir must be set when use_image=True")

        self.use_image = use_image
        self.use_logic_augmented_features = use_logic_augmented_features
        self.use_rule_components = use_rule_components
        self.use_clip_features = use_clip_features
        self.debug = debug
        self.image_dir = image_dir
        self.transform = transform
        self.clip_features_dir = None
        if self.use_clip_features:
            if 'train' in json_path:
                self.clip_features_dir = '/workspaces/SpLiCE/coco/clip_features/train'
            elif 'test' in json_path or 'val' in json_path:
                self.clip_features_dir = '/workspaces/SpLiCE/coco/clip_features/val'
            else:
                raise ValueError("Cannot determine split from json_path")
        self.items = []
        self.debug_count = 0

        for image_id, sample in data["images"].items():
            categories = sample.get("categories")
            labels = sample.get("labels")
            file_name = sample.get("file_name")
            rule_components = sample.get("rule_components", [[] for _ in range(10)])

            if categories is None or labels is None:
                continue

            self.items.append(
                {
                    "image_id": image_id,
                    "categories": torch.tensor(categories, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "rule_components": rule_components,
                    "file_name": file_name,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def encode_rule_components(self, rule_components):
        """
        Convert rule_components (list of lists) into a feature tensor.
        
        rule_components[i] is a list of positive component indices that matched for rule i.
        Returns a tensor of shape (10,) where each element indicates if at least one 
        component matched for that rule (binary: 1 if components matched, 0 otherwise).
        
        This essentially encodes whether each rule has at least one satisfied positive component.
        """
        component_features = torch.zeros(10, dtype=torch.float32)
        for rule_idx in range(10):
            # If the list is non-empty, at least one positive component matched
            if rule_idx < len(rule_components) and len(rule_components[rule_idx]) > 0:
                component_features[rule_idx] = 1.0
        return component_features

    # def augment_logic_features(self, counts_tensor):
    #     """
    #     Augment the input features with logic-derived features to help linear models
    #     handle non-linear rule conditions.

    #     counts_tensor: Tensor of shape (91,) with COCO category counts.
    #     Returns: Augmented tensor with additional logic features.
    #     """
    #     # COCO category indices (0-based)
    #     idx_person = 0
    #     idx_bicycle = 1
    #     idx_car = 2
    #     idx_motorcycle = 3
    #     idx_bus = 5
    #     idx_train = 6
    #     idx_truck = 7
    #     idx_traffic_light = 9
    #     idx_bottle = 39
    #     idx_cup = 41
    #     idx_bowl = 45
    #     idx_pizza = 53
    #     idx_dog = 16
    #     idx_cow = 19
    #     idx_sheep = 18
    #     idx_elephant = 20
    #     idx_surfboard = 37
    #     idx_chair = 56
    #     idx_couch = 57


    #     # 1. Binarized version of the entire input (presence indicators)
    #     binary_presence = (counts_tensor > 0).float()

    #     # 2. Comparative features for rules requiring differences/counts
    #     # Rule 6: Car Majority (more cars than trucks)
    #     car_diff = counts_tensor[idx_car] - counts_tensor[idx_truck]
    #     both_vehicles_present = (counts_tensor[idx_car] > 0) & (counts_tensor[idx_truck] > 0)
    #     car_majority = ((car_diff > 0) & both_vehicles_present).float()

    #     # Rule 10: Surf Trip (same number of persons and surfboards, both > 0)
    #     surf_diff = counts_tensor[idx_person] - counts_tensor[idx_surfboard]
    #     both_present = torch.logical_and(counts_tensor[idx_person] > 0, counts_tensor[idx_surfboard] > 0)
    #     surf_trip = torch.logical_and(surf_diff == 0, both_present).float()

    #     # 3. Aggregation features for exclusivity and counts
    #     # Rule 2: Double Serving (exactly two of bottle, cup, pizza)
    #     sum_food = binary_presence[idx_bottle] + binary_presence[idx_cup] + binary_presence[idx_pizza]
    #     double_serving = (sum_food == 2).float()

    #     # Rule 4: Dog or Car (exactly one of dog or car)
    #     sum_dog_car = binary_presence[idx_dog] + binary_presence[idx_car]
    #     dog_or_car = (sum_dog_car == 1).float()

    #     # Rule 5: Three of a kind (exactly three bowls or exactly three cups)
    #     three_bowls = (counts_tensor[idx_bowl] == 3).float()
    #     three_cups = (counts_tensor[idx_cup] == 3).float()
    #     three_of_kind = torch.max(three_bowls, three_cups)

    #     # Rule 8: Single Mode Traffic (exactly one of bicycle, motorcycle, car, bus)
    #     sum_traffic = binary_presence[idx_bicycle] + binary_presence[idx_motorcycle] + binary_presence[idx_car] + binary_presence[idx_bus]
    #     single_mode = (sum_traffic == 1).float()

    #     # Rule 9: Personal Transport (person and exactly one of bicycle or car)
    #     has_person = binary_presence[idx_person]
    #     sum_transport = binary_presence[idx_bicycle] + binary_presence[idx_car]
    #     personal_transport = (has_person.bool() & (sum_transport == 1)).float()

    #     # Rule 1: Signal and Ride (traffic light and exactly one of bicycle, bus, train)
    #     has_traffic = binary_presence[idx_traffic_light]
    #     sum_ride = binary_presence[idx_bicycle] + binary_presence[idx_bus] + binary_presence[idx_train]
    #     signal_ride = torch.logical_and(has_traffic.bool(), sum_ride == 1).float()

    #     # Rule 3: Herd Alone (two or more of same animal type, no person)
    #     no_person = ~binary_presence[idx_person].bool()
    #     two_cows = (counts_tensor[idx_cow] >= 2).float()
    #     two_elephants = (counts_tensor[idx_elephant] >= 2).float()
    #     two_sheep = (counts_tensor[idx_sheep] >= 2).float()
    #     herd_alone = torch.logical_and((two_cows + two_elephants + two_sheep) > 0, no_person).float()

    #     # Rule 7: Empty Seat (couch or chair, no person)
    #     has_furniture = (counts_tensor[idx_chair] > 0) | (counts_tensor[idx_couch] > 0)
    #     empty_seat = torch.logical_and(has_furniture, ~binary_presence[idx_person].bool()).float()

    #     # # Rule 7: Empty Seat (couch or chair, no person)
    #     # has_furniture = binary_presence[idx_chair] + binary_presence[idx_couch]
    #     # empty_seat = ((has_furniture > 0) & ~binary_presence[idx_person].bool()).float()

    #     # Concatenate all new features
    #     new_features = torch.stack([
    #         signal_ride,
    #         double_serving,
    #         herd_alone,
    #         dog_or_car,
    #         three_of_kind,
    #         car_majority,
    #         empty_seat,
    #         single_mode,
    #         personal_transport,
    #         surf_trip,
    #     ])

    #     # Return original counts + binary presence + new logic features
    #     return torch.cat([counts_tensor, binary_presence, new_features], dim=0)

    def augment_logic_features(self, counts_tensor):
        # DIE WAHREN INDIZES: Exaktes Mapping der raw COCO category_ids
        idx_person = 1
        idx_bicycle = 2
        idx_car = 3
        idx_motorcycle = 4
        idx_bus = 6
        idx_train = 7
        idx_truck = 8
        idx_traffic_light = 10
        
        idx_dog = 18
        idx_sheep = 20
        idx_cow = 21
        idx_elephant = 22
        
        idx_surfboard = 42
        idx_bottle = 44
        idx_cup = 47
        idx_bowl = 51
        idx_pizza = 59
        
        idx_chair = 62
        idx_couch = 63

        # Basis: Ist etwas vorhanden?
        binary_presence = (counts_tensor > 0).float()
        no_person = ~binary_presence[idx_person].bool()

        # Rule 1: Signal and Ride (Traffic light AND exactly one of bicycle, bus, train)
        has_traffic = binary_presence[idx_traffic_light].bool()
        sum_ride = binary_presence[idx_bicycle] + binary_presence[idx_bus] + binary_presence[idx_train]
        signal_ride = (has_traffic & (sum_ride == 1)).float()

        # Rule 2: Double Serving (Exactly two of bottle, cup, pizza)
        sum_food = binary_presence[idx_bottle] + binary_presence[idx_cup] + binary_presence[idx_pizza]
        double_serving = (sum_food == 2).float()

        # Rule 3: Herd Alone (2+ of same animal type, no person)
        has_herd = (counts_tensor[idx_cow] >= 2) | (counts_tensor[idx_elephant] >= 2) | (counts_tensor[idx_sheep] >= 2)
        herd_alone = (has_herd & no_person).float()

        # Rule 4: Dog or Car (XOR: exactly one of dog or car)
        sum_dog_car = binary_presence[idx_dog] + binary_presence[idx_car]
        dog_or_car = (sum_dog_car == 1).float()

        # Rule 5: Three of a kind (Exactly 3 bowls OR exactly 3 cups)
        three_bowls = counts_tensor[idx_bowl] == 3
        three_cups = counts_tensor[idx_cup] == 3
        three_of_kind = (three_bowls | three_cups).float()

        # Rule 6: Car Majority (Cars > trucks AND both appear at least once)
        car_majority = ((counts_tensor[idx_car] > counts_tensor[idx_truck]) & 
                        (counts_tensor[idx_car] > 0) & 
                        (counts_tensor[idx_truck] > 0)).float()

        # Rule 7: Empty Seat (Couch OR chair, AND no person)
        has_furniture = binary_presence[idx_chair].bool() | binary_presence[idx_couch].bool()
        empty_seat = (has_furniture & no_person).float()

        # Rule 8: Single Mode Traffic (Exactly ONE of bicycle, motorcycle, car, bus)
        sum_traffic = binary_presence[idx_bicycle] + binary_presence[idx_motorcycle] + binary_presence[idx_car] + binary_presence[idx_bus]
        single_mode = (sum_traffic == 1).float()

        # Rule 9: Personal Transport (Person AND exactly one of bicycle or car)
        sum_transport = binary_presence[idx_bicycle] + binary_presence[idx_car]
        personal_transport = (binary_presence[idx_person].bool() & (sum_transport == 1)).float()

        # Rule 10: Surf Trip (Same number of persons and surfboards, both > 0)
        surf_trip = ((counts_tensor[idx_person] == counts_tensor[idx_surfboard]) & 
                     (counts_tensor[idx_person] > 0)).float()

        # STACK IN EXAKTER REIHENFOLGE DER LABELS (Rule 1 bis 10)
        new_features = torch.stack([
            signal_ride,         # Label Index 0
            double_serving,      # Label Index 1
            herd_alone,          # Label Index 2
            dog_or_car,          # Label Index 3
            three_of_kind,       # Label Index 4
            car_majority,        # Label Index 5
            empty_seat,          # Label Index 6
            single_mode,         # Label Index 7
            personal_transport,  # Label Index 8
            surf_trip,           # Label Index 9
        ])

        return torch.cat([counts_tensor, new_features], dim=0)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        categories = item["categories"]
        labels = item["labels"]

        if self.use_clip_features:
            clip_file = os.path.join(self.clip_features_dir, f"{item['image_id']}.npy")
            categories = torch.from_numpy(np.load(clip_file)).float()
        elif self.use_logic_augmented_features:
            categories = self.augment_logic_features(categories)

        if self.use_rule_components:
            rule_comp_features = self.encode_rule_components(item["rule_components"])
            # Append rule component features to the input
            categories = torch.cat([categories, rule_comp_features], dim=0)

        if self.use_image:
            image_path = os.path.join(self.image_dir, item["file_name"])
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, categories, labels

        return categories, labels


def collate_cocologic(batch):
    """Collate function for CocoLogicDataset when returning only categories + labels."""
    categories = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    return categories, labels
# cd /workspaces/SpLiCE python coco/train_cocologic.py --model_type linear --use_rule_components --epochs 50