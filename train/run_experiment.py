from finetune_skin import train as skin_train
from finetune_UCLH import train as UCLH_train
from finetune_kits23 import train as kits23_train
from inference.inference_skin import inference as skin_inference
from inference.inference_UCLH import inference as UCLH_inference
from inference.inference_kits23 import inference as kits23_inference
import torch


def main():
    metrics = {
        'epoch': [],
        'test_top1': [],
        'test_recall': [],
        'test_auc': [],
    }
    # change dataset and model for finetune
    skin_data = "skin 224"
    skin_pretrain_model = r"D:\Msc project\code\checkpoint\skin\vit\best_skinmodel_acc.pth"
    skin_inference_model =  r"D:\Msc project\code\train\best_finetunemodel_acc_skin.pth"
    kits23_data = "2D Slices 224 Tumor"
    kits23_pretrain_model = r"D:\Msc project\code\checkpoint\kits23\vit\latest_kits23model.pth"
    kits23_inference_model = r"D:\Msc project\code\train\bestacc_kits23model_finetune.pth"
    UCLH_data = "UCLH_224_L3_3HU"
    UCLH_pretrain_model = r"D:\Msc project\code\checkpoint\UCLH\3 HU\resnet\latest_UCLHmodel_HU3.pth"
    UCLH_inference_model = r"D:\Msc project\code\train\best_finetunemodel_acc_UCLH_HU1.pth"
    image_encoder = 'resnet50d' #'vision transformer', 'resnet50d'

    for i in range(1,11):
        # change inference and train function for datasets
        UCLH_train(UCLH_data,image_encoder,UCLH_pretrain_model)
        val_recall, val_auc, top1_val = UCLH_inference(UCLH_data,image_encoder,UCLH_inference_model)

        def to_float(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().item()
            return float(x)

        metrics['epoch'].append(i)
        metrics['test_top1'].append(to_float(top1_val))
        metrics['test_recall'].append(to_float(val_recall))
        metrics['test_auc'].append(to_float(val_auc))

    # save training metrics into a csv file
    import pandas as pd
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("inference_UCLH.csv", index=False)
    print("Saved training metrics to csv")


if __name__ == "__main__":
    main()
