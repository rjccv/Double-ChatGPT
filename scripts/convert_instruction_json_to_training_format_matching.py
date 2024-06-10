import os
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--output_json_folder", required=True, help="")

    args = parser.parse_args()

    return args

def get_count_list(input_json_contents):
    i = 1
    listy = []
    img_id = input_json_contents[0]["ground_filename"]
    for content in input_json_contents[1:]:
        if img_id == content["ground_filename"]:
            i += 1
        else:
            listy.append(i)
            i = 1
        img_id = content["ground_filename"]
    return listy

def main():
    args = parse_args()
    # json_files = glob(args.input_json_file + '/*')
    # os.makedirs(args.output_json_folder, exist_ok=True)
    # output_json_folder = args.output_json_folder
    output_json_file = args.output_json_folder

    # if not os.path.isfile(output_json_file):
    #     # Create the file if it doesn't exist
    #     with open(output_json_file, 'w') as f:
    #         pass

    # output_json_contents = []
    # for json_file in json_files:
        # if "ground" in json_file or "scene_3" in json_file or "scene_6" in json_file:
        #     continue
    input_json_contents = json.load(open(args.input_json_file, 'r'))

    # input_json_contents = sorted(json_contents, key=sort_by_name) 
    output_json_contents = []

    for json_content in input_json_contents:
        img_id_1 = Path(json_content['aerial_filename']).stem
        img_id_2 = Path(json_content['ground_filename']).stem
        output_content = {'id1': img_id_1, 'img1': f"{img_id_1}.pkl", "id2": img_id_2, 'img2': f"{img_id_2}.pkl", 'conversations': []}      

        for key, val in json_content.items():
            if key.startswith("Q"):
                question = val
            if key.startswith("A"):
                answer = val
        output_content['conversations'].append({'from': 'human', 'value': (question).replace('\n<img2><mask2>', '').replace('<mask1>', '<img2>')})
        output_content['conversations'].append({'from': 'gpt', 'value': answer})

        output_json_contents.append(output_content)
        


    # output_json_file = os.path.join(output_json_folder, json_file.split('/')[-1])
    print(f"Total annotations retained: {len(output_json_contents)}")

    with open(output_json_file, 'w') as f:
        json.dump(output_json_contents, f, indent=4)


if __name__ == "__main__":
    main()

