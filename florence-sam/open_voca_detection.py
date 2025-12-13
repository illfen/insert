import argparse
from utils.sam import load_sam_image_model,run_sam_inference
from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
import supervision as sv
import torch
from PIL import Image
import numpy as np
import os

DEVICE = torch.device("cuda")

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


def annotate_image_box(image, detections):
    output_image = np.zeros_like(np.array(image))  

    for i, box in enumerate(detections.xyxy):  
        x1, y1, x2, y2 = map(int, box)
        output_image[y1:y2, x1:x2] = [255, 255, 255]
    
    return Image.fromarray(output_image)

def annote_image_mask_to_rgb(detections, image):
    mask = detections.mask 
    mask = mask[0]
    mask = (mask * 255).astype(np.uint8)

    if mask.shape != image.size[::-1]:  # image.size is (width, height)
        raise ValueError("Mask size must match image size.")
    
    image_np = np.array(image)
    result_image = np.zeros_like(image_np)
    result_image[mask == 255] = image_np[mask == 255]
    result_image_pil = Image.fromarray(result_image)
    
    return result_image_pil

def annote_image_mask( detections):
    mask = detections.mask 
    array = mask[0] * 255
    array = array.astype(np.uint8)
    return  Image.fromarray(array)


def run(text_input: str, image_input: Image.Image):
    texts = [prompt.strip() for prompt in text_input.split(",")]
    detections_list = []
    
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        # detections = {}
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        detections_list.append(detections)

    detections = sv.Detections.merge(detections_list)
    detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)

    return annotate_image(image_input, detections), annotate_image_box(image_input, detections),annote_image_mask(detections),annote_image_mask_to_rgb(detections,image_input)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script with text and path input")
    parser.add_argument("--text", type=str, required=True, help="Text input for the model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    
    text_input = args.text
    image_path = args.image_path

    print(f"Processing images in {image_path}...")
    print(f"dirs: {os.listdir(image_path)}")  
    for subdir, dirs, files in os.walk(image_path):
        print(f"Processing images in {subdir}...")
        print(f"dirs: {dirs}")
        if 'rgb' in dirs:
            rgb_dir = os.path.join(subdir, 'rgb')
            detection_dir = os.path.join(subdir, 'detection_florence2')
            
            if not os.path.exists(detection_dir):
                os.makedirs(detection_dir)
            for file_name in os.listdir(rgb_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    rgb_image_path = os.path.join(rgb_dir, file_name)
                    
                    image_input = Image.open(rgb_image_path)
                    _,detection, mask, mask_rgb = run(text_input, image_input) 
                    
                    detection_image_path = os.path.join(detection_dir, file_name)
                    detection.save(detection_image_path)
                    # 保存彩色标注图
                    # annotated_image, _, _, _ = run(text_input, image_input) 
                    # detection_image_path = os.path.join(detection_dir, file_name)
                    # annotated_image.save(detection_image_path)
            



