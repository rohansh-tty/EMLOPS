import math
import os 
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.catdog_classifier import CatDogClassifier
from src.utils.logging_utils import task_wrapper, get_rich_progress
from src.utils.split_dataset import prep_inference_data 

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_labels = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever', 'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier']
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_figures(figures, nrows = 1, ncols=1, output_file='output.png'):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    for fig, ax in zip(figures, axes.flat):
        if fig['image']:
            image = fig['image']
            title = f"Pred: {fig['confidence']}\n Label: {fig['label']}"
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
        else:
            # Remove extra axes (empty boxes)
            fig.delaxes(ax)  
    # plt.tight_layout() # optional
    plt.savefig(output_file, dpi=300)

# @task_wrapper
def main(input_folder, output_folder, ckpt_path):
    try:
        base_dir = Path(__file__).resolve().parent.parent / 'src'
        data_dir = base_dir / "data"
        source_dir = data_dir / "dataset"
         # add inference data
        infer_dataset_dir = input_folder #os.path.join(os.path.join(input_dir, 'data'), 'infer')
        if os.path.exists(infer_dataset_dir):
            if not os.listdir(infer_dataset_dir):
                prep_inference_data(source_dir, infer_dataset_dir)
       
       
        
        model = CatDogClassifier.load_from_checkpoint(ckpt_path)
        model.eval()
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        image_files = list(input_folder.glob('*'))
       
        pred_obj = [] # dict() # {"img": <Image> "label": 'Boxer', 'confidence': 0.09} 
        with get_rich_progress() as progress:
            task = progress.add_task("[green]Processing images...", total=len(image_files))
            for image_file in image_files:
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img, img_tensor = load_image(image_file)
                    predicted_label, confidence = infer(model, img_tensor.to(model.device))
                    
                    output_file = output_folder / f"{image_file.stem}_prediction.png"
                    save_prediction_image(img, predicted_label, confidence, output_file)
                    pred_obj.append({"image": img, "label": predicted_label, "confidence": round(100*confidence,2)})
                    progress.console.print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                    progress.advance(task)
        
        # aggregated plot of all predictions 
        col = 4 
        row = math.ceil(len(pred_obj)/4)
           
        output_file = output_folder / "final_prediction.png"
        plot_figures(pred_obj, nrows=row, ncols=col, output_file=output_file) 
    
    except Exception as e:
        print(f'Failed to run main () .. {e}')

if __name__ == "__main__":
    
    base_dir = Path(__file__).resolve().parent.parent / "src"
    input_dir = base_dir / "data" / "infer" 
    
    # create samples folder to store infernce images 
    os.makedirs(os.path.join(input_dir, 'samples'), exist_ok=True)
    infer_dataset_dir = base_dir / "data" / "infer" / "samples"
   
    # to store results 
    os.makedirs(os.path.join(input_dir, 'results'), exist_ok=True)
    output_dir = base_dir / "data" / "infer" / "results"
   
    # to store logs 
    os.makedirs(os.path.join(input_dir, 'logs'), exist_ok=True)
    log_dir = base_dir / "infer" / "logs"
    
    checkpoint_dir = base_dir / "checkpoints"
    checkpoints = [m for m in os.listdir(checkpoint_dir)]
    if checkpoints:
        try:
            model_file = sorted(checkpoints)[0] # choosing model wiht lowest loss
            ckpt_path = os.path.join(checkpoint_dir, model_file)
            # setup_logger(log_dir / "infer_log.log")
            main(infer_dataset_dir, output_dir, ckpt_path)
        except Exception as e:
            import traceback 
            print(f'Exception while inference {traceback.format_exc()}')