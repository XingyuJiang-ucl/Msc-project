import time
import torch.nn.functional as F
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from Dataset.Dataset import ImageDataset_1,ImageDataset_2
from model.finetune_attmodel import Net
from helper import dotdict
import torch
from torchvision import transforms
from pathlib import Path
import json
import torch.optim as optim
import torchmetrics
from torchmetrics.functional import mean_squared_error, accuracy,auroc,recall
from tqdm import tqdm


def load_tabular_data(json_file, keys_to_skip=None, device='cpu'):
    """
    Load tabular features  from a JSON file.

    Parameters:
        json_file (str): Path to the JSON file with records.
        keys_to_skip (list[str], optional): Field names to ignore. Defaults to None.
        device (str): Device for output tensors. Defaults to 'cpu'.

    Returns:
        data_dict (dict): Maps case_id to {'cat_tab': IntTensor, 'cont_tab': IntTensor}.
        targets_dict (dict): Maps case_id to one-hot FloatTensor of the prediction label.
    """
    if keys_to_skip is None:
        keys_to_skip = []

    with open(json_file, "r") as f:
        records = json.load(f)

    data_dict = {}
    # For counting unique values of each categorical feature: key -> set of unique values
    cat_unique = {}
    # Record the order of categorical keys (after removing keys_to_skip) from the first encountered record
    ordered_cat_keys = None
    # List to store the labels values from cat_tab for each record
    targets_dict = {}

    for record in records:
        case_id = record.get("case_id")
        if case_id is None:
            continue  # Skip records without a case_id

        cat_data = record.get("cat_tab", {}).copy()

        # Extract  "diagnostic" from cat_data and store it in targets_dict,
        if "diagnostic" in cat_data:
            target_value = cat_data["diagnostic"] - 1
            target_value = torch.tensor(target_value, dtype=torch.long, device=device)
            target_value = F.one_hot(target_value, num_classes=6).float()
            targets_dict[case_id] = torch.tensor(target_value, dtype=torch.float, device=device)
        else:
            targets_dict[case_id] = (torch.zeros(6, device=device))  # If key is missing, store default value -1

        # Process categorical data: copy and remove keys to skip
        for key in keys_to_skip:
            if key in cat_data:
                del cat_data[key]
        # Record the order of keys if not already recorded and if cat_data is not empty
        if ordered_cat_keys is None and cat_data:
            ordered_cat_keys = list(cat_data.keys())
        # Update the set of unique values for each categorical feature
        for key, value in cat_data.items():
            if key not in cat_unique:
                cat_unique[key] = set()
            cat_unique[key].add(value)
        # Convert categorical data to tensor using the original dictionary order
        cat_values = [cat_data[k] for k in cat_data.keys()]
        cat_tensor = torch.tensor(cat_values, dtype=torch.int, device=device)

        # Process continuous data: copy and remove keys_to_skip if needed
        cont_data = record.get("cont_tab", {}).copy()
        for key in keys_to_skip:
            if key in cont_data:
                del cont_data[key]

        # Convert continuous data to tensor using the original dictionary order
        cont_values = [cont_data[k] for k in cont_data.keys()]
        cont_tensor = torch.tensor(cont_values, dtype=torch.int, device=device)

        # Store the tensors in data_dict with case_id as the key
        data_dict[case_id] = {"cat_tab": cat_tensor, "cont_tab": cont_tensor}

    # # Convert the list of target values to a tensor
    # targets = torch.tensor(targets_dict, dtype=torch.float, device=device)

    return data_dict, targets_dict


def collate_fn(batch):
    images, case_ids = zip(*batch)
    images = torch.stack(images, 0)
    return images, case_ids


def calculate_tabular_embedding(model, batchsize, table_train):
    """
    Batch calculate tabular embeddings and return a dictionary with case_id as the key and the corresponding embeddings as the value.

    Parameters:
    model:
     A model that includes the forward tab function, where the forward tab of the model accepts a dictionary,
    The dictionary contains' cat_tab 'and' cont_tab ', and returns a batch of embedded tensors.
    batchsize:
     The number of samples calculated for each batch
    table_train: dict，
     The format is {case_id (int): {"cat_tab": cat_tensor, "cont_tab": cont_tenser}}

    return:
    embedding_dict: dict，
     The format is {case_id (int): embedding (tensor)}
    """
    embedding_dict = {}
    # Process in ascending order of case_id
    sorted_case_ids = sorted(table_train.keys())

    for i in range(0, len(sorted_case_ids), batchsize):
        #  case_id in current batch
        batch_keys = sorted_case_ids[i:i + batchsize]
        batch_cat_list = []
        batch_cont_list = []
        # Collect tensor data for each case in the current batch
        for case_id in batch_keys:
            sample = table_train[case_id]
            batch_cat_list.append(sample["cat_tab"])
            batch_cont_list.append(sample["cont_tab"])
        # stack as batch
        batch_cat = torch.stack(batch_cat_list, dim=0)
        batch_cont = torch.stack(batch_cont_list, dim=0)

        # create batch dict
        batch_data = {"cat_tab": batch_cat, "cont_tab": batch_cont}
        # Calculate embedding
        emb_batch = model.forward_tab(batch_data)

        # store embedding
        for j, case_id in enumerate(batch_keys):
            embedding_dict[case_id] = emb_batch[j]

    return embedding_dict


def train_one_epoch(model, train_loader, embedding_tab, labels,optimizer, scaler, DEVICE):
    """
    Trains the model for one epoch using mixed precision.

    Returns:
      epoch_loss: Average loss for the epoch.
      result of metrics: Average metrics value for the epoch
    """
    model.train()
    running_loss = 0.0
    total_samples = 0
    sum_top1 = 0.0
    sum_top3 = 0.0
    sum_auc = 0.0
    sum_recall = 0.0

    progress_bar = tqdm(total=len(train_loader), desc="Training", leave=False)
    for images, case_ids in train_loader:
        images = images.to(DEVICE)
        # Extract tabular embeddings for each case_id (create a new leaf tensor)
        batch_tab_emb_list = []
        batch_labels_list = []
        for cid in case_ids:
            tab_emb = embedding_tab[cid].detach().clone().requires_grad_().to(DEVICE)
            batch_tab_emb_list.append(tab_emb)
            batch_labels_list.append(labels[cid].argmax())
        batch_tab_emb = torch.stack(batch_tab_emb_list, dim=0)
        batch_labels = torch.stack(batch_labels_list, dim=0).to(DEVICE)
        #metric_labels = batch_labels * max_vitals

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            emb_image = model.forward_image(images)
            logits, loss = model.forward_loss(emb_image, batch_tab_emb,batch_labels)
        # results = logits * max_vitals
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # 使用 functional API 动态指定 num_classes 为当前 batch 的样本数
        # current_num_classes = images.size(0) # for fake label
        # current_num_classes = len(torch.unique(labels))
        # mseresult = mean_squared_error(results,metric_labels)
        acc1 = accuracy(logits, batch_labels, task="multiclass", average='micro',num_classes=6)
        acc3 = accuracy(logits, batch_labels, task="multiclass", top_k=3, average='micro',num_classes=6)
        auc = auroc(logits,batch_labels,task="multiclass", average='macro',num_classes=6)
        recall_value = recall(logits,batch_labels,task="multiclass", average='macro',num_classes=6)
        # sum_mse += mseresult * batch_size
        sum_top1 += acc1 * batch_size
        sum_top3 += acc3 * batch_size
        sum_auc += auc * batch_size
        sum_recall += recall_value * batch_size

        # progress_bar.set_postfix(loss=loss.item(), mse = mseresult.item())
        # Set tqdm bar
        progress_bar.set_postfix(loss=loss.item(),
                                 top1=acc1.item(),
                                 top3=acc3.item(),
                                 auc = auc.item(),
                                 recall = recall_value.item())
        progress_bar.update(1)
    progress_bar.close()

    epoch_loss = running_loss / total_samples
    # avg_mse = sum_mse / total_samples
    top1_acc = sum_top1 / total_samples
    top3_acc = sum_top3 / total_samples
    total_auc = sum_auc / total_samples
    total_recall = sum_recall / total_samples

    return epoch_loss,  top1_acc, top3_acc, total_auc,total_recall

def val_one_epoch(model, val_loader, embedding_tab, labels, DEVICE):
    """
    Validates the model for one epoch using mixed precision.

    Returns:
      epoch_loss: Average loss for the epoch.
      result of metrics: Average metrics value for the epoch
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    # max_vital_day = 7450
    # sum_mse = 0.0
    sum_top1 = 0.0
    sum_top3 = 0.0
    sum_auc = 0.0
    sum_recall = 0.0

    progress_bar = tqdm(total=len(val_loader), desc="Validation", leave=False)
    with torch.no_grad():
        for images, case_ids in val_loader:
            images = images.to(DEVICE)
            batch_tab_emb_list = []
            batch_labels_list = []
            for cid in case_ids:
                tab_emb = embedding_tab[cid].detach().clone().to(DEVICE)
                batch_tab_emb_list.append(tab_emb)
                batch_labels_list.append(labels[cid].argmax())
            batch_tab_emb = torch.stack(batch_tab_emb_list, dim=0)
            batch_labels = torch.stack(batch_labels_list, dim=0).to(DEVICE)
            # metric_labels = batch_labels * max_vital_day

            with torch.cuda.amp.autocast():
                emb_image = model.forward_image(images)
                logits,loss = model.forward_loss(emb_image, batch_tab_emb,batch_labels,output_type='loss')

            # results = logits * max_vital_day
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            acc1 = accuracy(logits, batch_labels, task="multiclass", average='micro', num_classes=6)
            acc3 = accuracy(logits, batch_labels, task="multiclass", top_k=3, average='micro', num_classes=6)
            auc = auroc(logits, batch_labels, task="multiclass", average='macro', num_classes=6)
            recall_value = recall(logits, batch_labels, task="multiclass", average='macro', num_classes=6)
            # sum_mse += mseresult * batch_size
            sum_top1 += acc1 * batch_size
            sum_top3 += acc3 * batch_size
            sum_auc += auc * batch_size
            sum_recall += recall_value * batch_size

            # progress_bar.set_postfix(mse=mseresult.item())
            progress_bar.set_postfix(loss=loss.item(),
                                     top1=acc1.item(),
                                     top3=acc3.item(),
                                     auc = auc.item(),
                                     recall = recall_value.item())
            progress_bar.update(1)
    progress_bar.close()

    epoch_loss = running_loss / total_samples
    top1_acc = sum_top1 / total_samples
    top3_acc = sum_top3 / total_samples
    top_auc = sum_auc / total_samples
    total_recall = sum_recall / total_samples

    return epoch_loss, top1_acc, top3_acc,top_auc, total_recall

def inference(data_root = "skin 224",model_name = 'vision transformer',weight_root = r"D:\Msc project\code\checkpoint\skin\vit\best_finetunemodel_acc_skin.pth"):
    """
    training pipeline.

    Parameters:
    data_root: folder name which store training and validation data
    model_name: name of image encoder, vision transformer or resnet50d
    weight_root: path of pretrain weight

    """
    current_dir = Path(__file__).resolve().parent
    data_root = current_dir.parent / data_root
    keys_to_skip = {"case_id","patient_id",'diagnostic'}

    # Create root
    train_image_root = data_root/ "train"
    train_table_root = data_root/ "train"/ "train.json"
    # Load training data
    table_train,label_train = load_tabular_data(train_table_root,keys_to_skip)
    n_cont_features = len(table_train['PAT_8_15_820.png']["cont_tab"])


    # Create root
    test_image_root = data_root / "test"
    test_table_root = data_root / "test" / "test.json"
    # Load test data
    table_test ,label_test= load_tabular_data(test_table_root, keys_to_skip)

    # Image augmentations for validation
    transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    batchsizetest = 128
    EPOCHS        = 1
    DEVICE = torch.device('cuda')

    # for batch_idx, (images, case_ids) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print("Images tensor shape:", images.shape)
    #     print("Case IDs:", case_ids)

    dataset_test= ImageDataset_2(test_image_root, transform=transform_test)
    print("Total number of test images:", len(dataset_test))
    test_loader = DataLoader(
        dataset_test,
        batch_size= batchsizetest,
        # num_workers=8,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,  # Ensure the data order remains consistent
        collate_fn=collate_fn
    )

    cat_dict = {'smoke': 3, 'drink': 3, 'background_father': 14, 'background_mother': 12, 'pesticide': 3, 'gender': 3,
     'skin_cancer_history': 3, 'cancer_history': 3, 'has_piped_water': 3, 'has_sewage_system': 3, 'fitspatrick': 7,
     'region': 15,  'itch': 4, 'grew': 4, 'hurt': 4, 'changed': 4, 'bleed': 4, 'elevation': 4,'biopsed': 3}
    # 'diagnostic': 7,}
    cat_cardinalities = []
    for feat, cnt in cat_dict.items():
        cat_cardinalities.append(cnt)

    cfg = dotdict(
        n_cont_features = n_cont_features,
        cat_cardinalities=cat_cardinalities,
        arch = model_name,
        d_block = 512,
        num_classes=6,
        att = True,
        img_dim=448 if model_name == 'vision transformer' else 2048  # vit 22M:576,vit 1M:448,resnet:2048
    )

    model = Net(pretrained=False, cfg=cfg).to(DEVICE)
    pretrained_dict = torch.load(weight_root,weights_only= True)
    model.load_state_dict(pretrained_dict,strict=False)

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        print(f"\n===== EPOCH {epoch}/{EPOCHS} =====")

        # Calculate tabular embedding
        model.eval()
        embedding_tab_test = calculate_tabular_embedding(model, batchsizetest, table_test)

        # validation
        test_loss, top1_test, top3_test, test_auc,test_recall = val_one_epoch(model, test_loader, embedding_tab_test, label_test, DEVICE)

        elapsed = time.time() - start_time
        print(f"Test Recall: {test_recall:.4f} | Test AUC: {test_auc:.4f} | Test Acc (Top1): {top1_test:.4f} | Test Acc (Top3): {top3_test:.4f}")

    return test_recall,test_auc, top1_test


if __name__ == '__main__':
    inference()