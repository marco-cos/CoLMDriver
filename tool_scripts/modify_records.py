import json
from pathlib import Path
import os
import numpy as np

def parse_broken_json(file_path):
    with open(file_path, "r") as f:
        content = f.read()  # Read the entire file content

    # Initialize the parsing result
    objects = []
    stack = []  # Stack for matching curly braces
    start_index = 0  # Starting index for the current object

    # Iterate over the file content and match curly braces
    for i, char in enumerate(content):
        if char == "{":
            stack.append(i)  # Record the position of the opening brace
        elif char == "}":
            if stack:
                stack.pop()  # Pop the most recent opening brace
                if not stack:  # If the stack is empty, a complete JSON object is identified
                    # Extract the object string
                    obj_str = content[start_index:i+1]
                    try:
                        obj = json.loads(obj_str)  # Parse as a JSON object
                        objects.append(obj)  # Add to the results list
                    except json.JSONDecodeError:
                        print(f"Unparseable object: {obj_str}")
                    start_index = i + 1  # Update the start index for the next object

    # Return the parsed objects as a valid JSON structure
    return objects

# Example: parsing a broken JSON file
route_b7_dict = {}
route_carnum_dict = {
    "r1_town05_ins_c": 2,
    "r2_town05_ins_c": 2,
    "r3_town05_ins_c": 2,
    "r4_town06_ins_c": 2,
    "r5_town06_ins_c": 2,
    "r6_town07_ins_c": 2,
    "r7_town05_ins_ss": 2,
    "r8_town05_ins_ss": 2,
    "r9_town06_ins_ss": 2,
    "r10_town07_ins_ss": 2,
    "r11_town05_ins_sl": 2,
    "r12_town06_ins_sl": 2,
    "r13_town05_ins_sl": 2,
    "r14_town07_ins_sl": 2,
    "r15_town07_ins_sl": 2,
    "r16_town05_ins_sl": 2,
    "r17_town05_ins_sr": 2,
    "r18_town05_ins_sr": 2,
    "r19_town05_ins_sr": 2,
    "r20_town06_ins_sr": 2,
    "r21_town07_ins_sr": 2,
    "r22_town07_ins_sr": 2,
    "r23_town05_ins_oppo": 3,
    "r24_town05_ins_rl": 3,
    "r25_town05_ins_crosschange": 3,
    "r26_town05_ins_chaos": 6,
    "r27_town06_hw_merge": 3,
    "r28_town06_hw_c": 6,
    "r29_town06_hw_merge": 4,
    "r30_town06_hw_merge": 4,
    "r31_town05_ins_oppo": 4,
    "r32_town05_ins_oppo": 4,
    "r33_town05_ins_rl": 4,
    "r34_town05_ins_rl": 4,
    "r35_town05_ins_crosschange": 4,
    "r36_town05_ins_crosschange": 4,
    "r37_town05_ins_chaos": 8,
    "r38_town05_ins_chaos": 8,
    "r39_town06_hw_c": 8,
    "r40_town06_hw_c": 8,
    "r41_town05_ins_oppo": 4,
    "r42_town05_ins_rl": 4,
    "r43_town05_ins_crosschange": 4,
    "r44_town05_ins_chaos": 8,
    "r45_town06_hw_merge": 4,
    "r46_town06_hw_c": 7
}

root_dir = "/GPFS/rhome/zijunwang/WorkSpace/V2Xverse/results/results_driving_0307_group_new"
prompt_material = []

# for i in range(3):
    # v2x_final_path = Path(root_dir+str(i)) / "image/v2x_final"
v2x_final_path = Path(root_dir) / "image/v2x_final"
for route in os.listdir(v2x_final_path):
    # route_b7_dict[route] = 0
    for exp in os.listdir(v2x_final_path / route):
        try:
            json_path = v2x_final_path / route / exp / 'log/group_negotiation_records.json'
            records = parse_broken_json(json_path)
            for record in records:
                for step, step_record in record.items():
                    for r, round_record in step_record.items():
                        if round_record["scores"]["total_score"] <= 7:
                            continue
                        # route_b7_dict[route] += 1
                        for key, value in round_record.items():
                            if key == "scores":
                                continue
                            # if not isinstance(value["prompt"], str) or not isinstance(value["response"], str) or not isinstance(value):
                            #     continue
                            
                            try:
                                if key != 'comm':
                                    # x = np.random.rand()
                                    # if x > 1/3:
                                    #     continue
                                    message_record = {}
                                    message_record["messages"] = []
                                    message_record["messages"].append({"role": "user", "content": value["prompt"]})
                                    message_record["messages"].append({"role": "assistant", "content": value["response"]}) 
                                    prompt_material.append(message_record)
                                else:
                                    for item in value:
                                        message_record = {}
                                        message_record["messages"] = []
                                        message_record["messages"].append({"role": "user", "content": item["prompt"]})
                                        message_record["messages"].append({"role": "assistant", "content": item["response"]})
                                        prompt_material.append(message_record)
                                        
                            except Exception as e:
                                pass
        except Exception as e:
            print(e)
print("Dataset size:", len(prompt_material))
# for key,value in route_b7_dict.items():
#     route_b7_dict[key] = value / route_carnum_dict[key]
# sorted_dict = dict(sorted(route_b7_dict.items(), key=lambda item: item[1], reverse=True))
# print(sorted_dict)

with open("prompt_material_bigger7_correct.json", 'w') as f:
    json.dump(prompt_material, f, indent=4)
