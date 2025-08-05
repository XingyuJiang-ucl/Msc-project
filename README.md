# Msc project - Xinyu Jiang - Multimodal AI for predicting post-surgical complications 

This is the source code for Xingyu Jiang's personal project in MSc Artifical Intelligence and Medical Imaging from University College London. The topic is 'Multimodal AI for predicting post-surgical complications'. 

To use pretrain weight of TinyVit, download from their offical implementation: [link](https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth).

For finetune and inference, run [run_experiment.py](train/run_experiment.py) after pretrain.



### To use attention module in finetune, there are two change need to do:

1. Delete or annotate line 583 in [tiny_vit.py](model/tiny_vit.py), use sequence before meanpooling.

2. After installing rtdl_revisiting_models, delete or annotate line 669 in rtdl_revisiting_models.py, use the whole sequence but not CLS token.
