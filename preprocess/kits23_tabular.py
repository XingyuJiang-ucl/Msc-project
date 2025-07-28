import json
import random

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dictionaries into dot-separated keys.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def detect_feature_types(records):
    """
    Determine types and collect values for each feature.
    """
    from collections import defaultdict
    values = defaultdict(list)
    for record in records:
        for key, val in record.items():
            values[key].append(val)

    types = {}
    for feature, vals in values.items():
        non_null = [v for v in vals if v is not None]
        if not non_null:
            types[feature] = 'unknown'
        elif all(isinstance(v, bool) for v in non_null):
            types[feature] = 'categorical_bool'
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
            types[feature] = 'continuous'
        elif all(isinstance(v, str) for v in non_null):
            types[feature] = 'categorical_str'
        else:
            types[feature] = 'unknown'
    return types, values


def build_mappings(types, values):
    """
    Create integer mappings for categorical features.
    """
    maps = {}
    for feature, ftype in types.items():
        if ftype == 'categorical_bool':
            maps[feature] = {False: 0, True: 1}
        elif ftype == 'categorical_str':
            unique_vals = sorted(v for v in set(values[feature]) if v is not None)
            maps[feature] = {v: i for i, v in enumerate(unique_vals)}
    return maps


def compute_cont_stats(types, values):
    """
    Compute min and max for continuous features.
    """
    stats = {}
    for feature, ftype in types.items():
        if ftype == 'continuous':
            non_null = [v for v in values[feature] if isinstance(v, (int, float))]
            if non_null:
                stats[feature] = (min(non_null), max(non_null))
    return stats

def transform(records, types, maps, stats):
    """
    Transform records:
    - Impute missing continuous features with -1 and categorical features with 0.
    - Normalize non-missing continuous features to [0,1].
    - Encode non-missing categorical features to integers.
    """
    transformed = []
    for rec in records:
        flat = {}
        for feat, val in rec.items():
            ftype = types.get(feat, 'unknown')

            if ftype == 'continuous':
                if val is not None:
                    min_v, max_v = stats.get(feat, (0, 0))
                    if max_v != min_v:
                        flat[feat] = (val - min_v) / (max_v - min_v)
                    else:
                        flat[feat] = 0.0
                else:
                    flat[feat] = -1.0
            elif ftype in ('categorical_bool', 'categorical_str'):
                flat[feat] = maps.get(feat, {}).get(val, 0)

            else:
                flat[feat] = -1.0 if val is None else val

        transformed.append(flat)
    return transformed


def main():
    """
    Load data, process features, split into splits, save JSON files.
    """
    with open('kits23.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('kits23img_number.json', 'r', encoding='utf-8') as f:
        images = json.load(f)

    flat = [flatten_dict(r) for r in data]
    types, values = detect_feature_types(flat)
    maps = build_mappings(types, values)
    stats = compute_cont_stats(types, values)

    transformed = transform(flat, types, maps, stats)

    train_img, val_img, test_img = [], [], []
    train_tab, val_tab, test_tab = [], [], []

    for img in images:
        cid = img['case_id']
        if 400 <= cid <= 488:
            test_img.append(img)
        elif 311 <= cid <= 399:
            val_img.append(img)
        else:
            train_img.append(img)

    for rec in transformed:
        cid = rec.get('case_id')
        cont = {f: rec[f] for f, t in types.items() if t == 'continuous'}
        cat = {f: rec[f] for f, t in types.items() if t in ('categorical_bool', 'categorical_str')}
        entry = {'case_id': cid, 'cat_tab': cat, 'cont_tab': cont}
        if 400 <= cid <= 488:
            test_tab.append(entry)
        elif 311 <= cid <= 399:
            val_tab.append(entry)
        else:
            train_tab.append(entry)

    with open('image_data_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_img, f, indent=2)
    with open('image_data_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_img, f, indent=2)
    with open('image_data_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_img, f, indent=2)

    with open('table_data_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_tab, f, indent=2)
    with open('table_data_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_tab, f, indent=2)
    with open('table_data_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_tab, f, indent=2)


if __name__ == '__main__':
    main()