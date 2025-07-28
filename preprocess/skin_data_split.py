import os
import json
import random
import shutil

def split_images_and_data(
    img_dir: str,
    table_json: str,
    output_val_json: str,
    output_test_json: str,
    output_train_json: str,
    val_img_dir: str,
    test_img_dir: str
):
    """
    Read all image files from img_dir, sort them by filename, then:
      - Use the first half as the validation set (val)
      - Use the second half as the test set (test)
    Match corresponding table records (from table_json) using case_id,
    and save them into three JSON files: output_val_json, output_test_json, output_train_json

    Also compute and print the number of categories for each categorical feature (i.e., one-hot vector length).
    """
    # List and sort all image files
    images = sorted(
        f for f in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, f))
    )
    if not images:
        raise ValueError(f"No images found in {img_dir}")

    # Split in order
    mid = len(images) // 2
    val_imgs = images[:mid]
    test_imgs = images[mid:]

    # Create output directories
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    # Copy validation and test images
    for fn in val_imgs:
        shutil.copy2(os.path.join(img_dir, fn), os.path.join(val_img_dir, fn))
    for fn in test_imgs:
        shutil.copy2(os.path.join(img_dir, fn), os.path.join(test_img_dir, fn))

    # Load JSON table data
    with open(table_json, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # Split records based on case_id
    val_set = set(val_imgs)
    test_set = set(test_imgs)

    val_records = []
    test_records = []
    train_records = []

    for r in records:
        cid = r.get('case_id')
        if cid in val_set:
            val_records.append(r)
        elif cid in test_set:
            test_records.append(r)
        else:
            train_records.append(r)

    # Save the three JSON files
    with open(output_val_json, 'w', encoding='utf-8') as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    with open(output_test_json, 'w', encoding='utf-8') as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
    with open(output_train_json, 'w', encoding='utf-8') as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"Validation set: {len(val_imgs)} images, {len(val_records)} records")
    print(f"Test set: {len(test_imgs)} images, {len(test_records)} records")
    print(f"Training set: {len(train_records)} records (not assigned to val/test)")


if __name__ == "__main__":
    split_images_and_data(
        img_dir=r"D:\Msc project\skin lesion\resized_224\imgs_part_2",
        table_json="output.json",
        output_val_json="val.json",
        output_test_json="test.json",
        output_train_json="train.json",
        val_img_dir="val",
        test_img_dir="test",
    )
