from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
import cv2
import numpy as np
import re
import csv
import argparse
import torch
torch.cuda.empty_cache()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images with Molmo and extract points.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate points.")
    parser.add_argument("--is_bd_box", action="store_true", help="Generate bounding box instead of points.")
    return parser.parse_args()



def extract_points(molmo_output, image_w, image_h):
    all_points = []
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points

def draw_points_on_image(image, points):

    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    return image


def save_image_with_points(original_image_path, points):


    image = Image.open(original_image_path)

    image_with_points = draw_points_on_image(image, points)


    image_with_points_pil = Image.fromarray(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))

    original_folder = os.path.dirname(original_image_path)
    
    if args.is_bd_box:
        points_folder = os.path.join(original_folder, 'bounding_box')
    else:
        points_folder = os.path.join(original_folder, 'points')
    if not os.path.exists(points_folder):
        os.makedirs(points_folder)

    original_filename = os.path.basename(original_image_path)

    new_image_path = os.path.join(points_folder, original_filename)

    image_with_points_pil.save(new_image_path)
    print(f"Image saved to: {new_image_path}")

args = parse_arguments()


# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-O-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
def save_image_with_mask(original_image_path, all_points):
    import cv2
    import os
    import numpy as np
    from PIL import Image

    image_shape = (512, 512)

    points = np.array(all_points, dtype=np.float32)

    binary_image = np.zeros(image_shape, dtype=np.uint8)

    if len(points) == 1:
        x, y = points[0]
        top_left = (int(x) - 25, int(y) - 25)
        bottom_right = (int(x) + 25, int(y) + 25)

        cv2.rectangle(binary_image, top_left, bottom_right, 1, thickness=-1)
    else:
        rect = cv2.minAreaRect(points)

        box = cv2.boxPoints(rect)  # box is float
        box = box.astype(np.int32)

        cv2.fillPoly(binary_image, [box], 1)

    binary_image_rgb = cv2.merge([binary_image * 255] * 3)

    original_folder = os.path.dirname(original_image_path)

    if not args.is_bd_box:
        masks_folder = os.path.join(original_folder, 'mask')
    else:
        masks_folder = os.path.join(original_folder, 'bounding_box_mask')
    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)

    original_filename = os.path.basename(original_image_path)

    new_image_path = os.path.join(masks_folder, original_filename)

    Image.fromarray(binary_image_rgb).save(new_image_path)
    print(f"Image saved to: {new_image_path}")

def get_points_for_image(image_path,prompt):
    # load the image
    image = Image.open(image_path)

    # process the image and text
    inputs = processor.process(
        images=[image],
        text=prompt
    )
    # Point the border of the areas for doll to wear a sunglasses

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    all_points = extract_points(generated_text, 512, 512)
    return all_points

folder_path = args.folder
output_csv = os.path.join(folder_path, "points.csv")


with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "point_x", "point_y"])
    
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_points = get_points_for_image(image_path,args.prompt)
            if len(all_points) == 0:
                print("Error: points is empty!")
            else:
                save_image_with_points(image_path, all_points)
                save_image_with_mask(image_path, all_points)
                for point in all_points:
                    writer.writerow([filename, point[0], point[1]])
    else:
        print(f"Skipping {filename}")
    print(f"Data has been saved in {output_csv}")


