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
        if json_content.get("Q1", None) is not None:           
            img_id_1 = Path(json_content['aerial_filename']).stem
            img_id_2 = Path(json_content['ground_filename']).stem
            output_content = {'id1': img_id_1, 'img1': f"{img_id_1}.pkl", "id2": img_id_2, 'img2': f"{img_id_2}.pkl", 'conversations': []}
            q1 = json_content["Q1"]
            output_content['conversations'].append({'from': 'human', 'value': f"<img1>\n{q1}\n<img2>"})
            output_content['conversations'].append({'from': 'gpt', 'value': json_content["A1"]})
        if json_content.get("Q2", None) is not None:
            output_content['conversations'].append({'from': 'human', 'value': (json_content["Q2"].split("<mask2>")[0]).replace('<mask1>', '').replace('<mask2>', '')})
            output_content['conversations'].append({'from': 'gpt', 'value': json_content["A2"]})
        if json_content.get("Q2", None) is not None:
            output_content['conversations'].append({'from': 'human', 'value': (json_content["Q3"]).replace('<mask1>', '').replace('<mask2>', '')})
            output_content['conversations'].append({'from': 'gpt', 'value': json_content["A3"]})

        output_json_contents.append(output_content)
        


    # output_json_file = os.path.join(output_json_folder, json_file.split('/')[-1])
    print(f"Total annotations retained: {len(output_json_contents)}")

    with open(output_json_file, 'w') as f:
        json.dump(output_json_contents, f, indent=4)

if __name__ == "__main__":
    main()

