import json
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt

def parse_broken_json(file_path):
    # Read the entire file content
    with open(file_path, "r") as f:
        content = f.read()

    # Initialize the result for parsing
    objects = []
    stack = []  # Used to match curly braces
    start_index = 0  # Starting index for the current object

    # Iterate over the file content to match curly braces
    for i, char in enumerate(content):
        if char == "{":
            stack.append(i)  # Record the position of the left curly brace
        elif char == "}":
            if stack:
                stack.pop()  # Pop the most recent left curly brace position
                if not stack:  # If the stack is empty, a complete JSON object has been found
                    # Extract object content
                    obj_str = content[start_index:i+1]
                    try:
                        obj = json.loads(obj_str)  # Parse to JSON object
                        objects.append(obj)  # Add to the result list
                    except json.JSONDecodeError:
                        print(f"Unable to parse object: {obj_str}")
                    start_index = i + 1  # Update the starting index for the next object

    # Return the parsed objects as a valid JSON structure
    return objects

# Example: Parse a corrupted JSON file

def date_to_second(exp):
    t = exp.split("_")[-5:]
    sign = int(t[-1]) + int(t[-2]) * 60 + int(t[-3]) * 3600 + int(t[-4]) * 3600 * 24 + int(t[-5]) * 3600 * 24 * 50
    # Here, a month is approximated by 50 days so that comparisons don't need to consider actual month lengths
    return sign

# root_dir = "/GPFS/rhome/zijunwang/WorkSpace/V2Xverse/results/results_driving_0306_group_record"
root_dir = "/GPFS/rhome/zijunwang/WorkSpace/V2Xverse/results/results_driving_0307_group_nofb"
save_dir = Path("vis_results/score_round_figures/nofb")
save_dir.mkdir(parents=True, exist_ok=True)
route_score_dict = {}
# for i in range(3):
#     v2x_final_path = Path(root_dir+str(i)) / "image/v2x_final"
v2x_final_path = Path(root_dir) / "image/v2x_final"
for route in os.listdir(v2x_final_path):
    score_array_all = []
    # route_b7_dict[route] = 0
    exp_name_list = os.listdir(v2x_final_path / route)
    latest_exp = sorted(exp_name_list, key=date_to_second)[-1]
    try:
        json_path = v2x_final_path / route / latest_exp / 'log/group_negotiation_records.json'
        records = parse_broken_json(json_path)
        for record in records:
            assert len(record) == 1
            for step, step_record in record.items():
                score_array = []
                if len(step_record) != 5:
                    continue
                for r, round_record in step_record.items():
                    eff_score = round_record["scores"]["efficiency_score"]
                    safety_score = round_record["scores"]["safety_score"]
                    total_score = 3 * eff_score + 7 * safety_score
                    score_array.append(total_score)
                score_array_all.append(score_array)
                    
    except Exception as e:
        print(e)
    if len(score_array_all) > 0:
        route_score_dict[route] = np.array(score_array_all)
x = [1, 2, 3, 4, 5]
final_score_array = []
for route, score_array in route_score_dict.items():
    average_score_round = np.mean(score_array, axis=0)
    final_score_array.append(score_array)
    print(route, average_score_round)
    plt.figure()
    plt.plot(x, average_score_round, marker='.')
    plt.xticks(x)
    plt.title(route)
    plt.xlabel("round")
    plt.ylabel("total score")
    plt.ylim(0, 10)
    plt.savefig(save_dir / f"{route}.png")
    plt.close()

final_score_array = np.concatenate(final_score_array, axis=0)
final_score_array = np.mean(final_score_array, axis=0)
print("average score:", final_score_array)
plt.figure()
plt.plot(x, final_score_array, marker='.')
plt.xticks(x)
plt.title("average score to round")
plt.xlabel("round")
plt.ylabel("total score")
plt.ylim(0, 10)
plt.savefig(save_dir / "average_score.png")
plt.close()
        

# for key,value in route_b7_dict.items():
#     route_b7_dict[key] = value / route_carnum_dict[key]
# sorted_dict = dict(sorted(route_b7_dict.items(), key=lambda item: item[1], reverse=True))
# print(sorted_dict)

# with open("prompt_material_bigger7.json", 'w') as f:
#     json.dump(prompt_material, f, indent=4)
