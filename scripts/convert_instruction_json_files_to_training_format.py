import os
import json
import argparse
from glob import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--output_json_folder", required=True, help="")

    args = parser.parse_args()

    return args


# def main():
#     args = parse_args()
#     json_files = glob(args.input_json_file + '/*')
#     # os.makedirs(args.output_json_folder, exist_ok=True)
#     # output_json_folder = args.output_json_folder
#     output_json_file = args.output_json_folder

#     # if not os.path.isfile(output_json_file):
#     #     # Create the file if it doesn't exist
#     #     with open(output_json_file, 'w') as f:
#     #         pass



#     output_json_contents = []
#     i = 0
#     for json_file in json_files:
#         if "scene_3" in json_file or "scene_6" in json_file:
#             continue
#         input_json_contents = json.load(open(json_file, 'r'))
#         # output_json_contents = []
#         for json_content in input_json_contents:
#             video_id = Path(json_content[-1]['filename']).stem
            
#             for content in json_content[:-1]:
#                 output_content = {'id': video_id, 'video': f"{video_id}.pkl", 'conversations': []}

#                 if i % 2 == 0:
#                     output_content['conversations'].append({'from': 'human', 'value': f"{content['Q']}\n<video>"})
#                 else:
#                     output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{content['Q']}"})
#                 i += 1

#                 output_content['conversations'].append({'from': 'gpt', 'value': content['A']})
#                 output_json_contents.append(output_content)

#         # output_json_file = os.path.join(output_json_folder, json_file.split('/')[-1])
#         print(f"Total annotations retained: {len(output_json_contents)}")

#     with open(output_json_file, 'w') as f:
#         json.dump(output_json_contents, f, indent=4)



def main():
    args = parse_args()
    # os.makedirs(args.output_json_folder, exist_ok=True)
    # output_json_folder = args.output_json_folder
    output_json_file = args.output_json_folder

    with open(args.input_json_file, "r") as f:
        input_json_contents = json.load(f)

    

    output_json_contents = []
    i = 0

    # output_json_contents = []
    for json_content in input_json_contents:
        video_id = Path(json_content[-1]['filename']).stem
        
        for content in json_content[:-1]:
            output_content = {'id': video_id, 'img': f"{video_id}.pkl", 'conversations': []}

            if i % 2 == 0:
                output_content['conversations'].append({'from': 'human', 'value': f"{content['Q']}\n<img>"})
            else:
                output_content['conversations'].append({'from': 'human', 'value': f"<img>\n{content['Q']}"})
            i += 1

            output_content['conversations'].append({'from': 'gpt', 'value': content['A']})
            output_json_contents.append(output_content)

    # output_json_file = os.path.join(output_json_folder, json_file.split('/')[-1])
    print(f"Total annotations retained: {len(output_json_contents)}")

    with open(output_json_file, 'w') as f:
        json.dump(output_json_contents, f, indent=4)

if __name__ == "__main__":
    main()

