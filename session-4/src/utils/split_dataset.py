import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, output_dir, val_split=0.2):
    # Create train/val directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    print(f'source : {source_dir}')
    print(f'source ls : {os.listdir(source_dir)}')
    print(f'output :{output_dir}') 
    # For each breed folder
    for breed in os.listdir(source_dir):
        breed_path = os.path.join(source_dir, breed)
        if not os.path.isdir(breed_path):
            continue
            
        # Get all images
        images = [f for f in os.listdir(breed_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f'Breed Path: {breed_path}, Image Len >> {len(images)}') 
        # Split into train/val
        train_imgs, test_imgs = train_test_split(images, test_size=0.5, random_state=42)
        test_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=91) 
        # Create breed folders in train/val
        os.makedirs(os.path.join(output_dir, 'train', breed), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', breed), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', breed), exist_ok=True)
         
        # Copy images
        for img in train_imgs:
            shutil.copy2(
                os.path.join(source_dir, breed, img),
                os.path.join(output_dir, 'train', breed, img)
            )
        
        for img in val_imgs:
            shutil.copy2(
                os.path.join(source_dir, breed, img),
                os.path.join(output_dir, 'val', breed, img)
            )
 
        for img in test_imgs:
            shutil.copy2(
                os.path.join(source_dir, breed, img),
                os.path.join(output_dir, 'test', breed, img)
            )


# Usage
# split_dataset(
#     source_dir='original_dataset',  # Your original dataset with breed folders
#     output_dir='dataset_root',      # Where to create the train/val split
#     val_split=0.2                   # 20% for validation
# )