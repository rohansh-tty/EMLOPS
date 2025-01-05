import os
import random
import shutil
from sklearn.model_selection import train_test_split

def prep_inference_data(source_dir, output_dir):
    try:
        random_idx = [random.randint(1,100) for i in range(10)]
        infer_imgs = []
        for i, breed in zip(random_idx, os.listdir(source_dir)):
            breed_path = os.path.join(source_dir, breed)
            if os.path.isdir(breed_path):
                images = [f for f in os.listdir(breed_path) if f.endswith(('.jpg', '.jpeg', '.png'))]    
                random_image_path = os.path.join(breed_path, images[i]) 
                infer_imgs.append(random_image_path)
        for img in infer_imgs:
            shutil.copy2(
                    img,
                    output_dir)
    except Exception as e:
        print(f'Failed to run inference {e}')        
            
def split_dataset(source_dir, output_dir, val_split=0.2):
    # Create train/val directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'infer'), exist_ok=True)
    
    # For each breed folder
    for breed in os.listdir(source_dir):
        breed_path = os.path.join(source_dir, breed)
        if not os.path.isdir(breed_path):
            continue
            
        # Get all images
        images = [f for f in os.listdir(breed_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        # Split into train/val
        train_imgs, _ = train_test_split(images, test_size=0.5, random_state=42)
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
                os.path.join(output_dir, 'test', breed, img))
            
            
   