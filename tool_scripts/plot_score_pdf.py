import json
from pathlib import Path
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image

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

def parse_broken_json(file_path):
    with open(file_path, "r") as f:
        content = f.read()  # Read the entire file content

    # Initialize the list to store parsed JSON objects
    objects = []
    stack = []  # Used to match curly braces
    start_index = 0  # Start index of the current JSON object

    # Traverse content to match curly braces
    for i, char in enumerate(content):
        if char == "{":
            stack.append(i)  # Record the position of the opening brace
        elif char == "}":
            if stack:
                stack.pop()  # Remove the latest opening brace position
                if not stack:  # Indicates a complete JSON object when stack is empty
                    # Extract the object string
                    obj_str = content[start_index:i+1]
                    try:
                        obj = json.loads(obj_str)  # Parse as JSON object
                        objects.append(obj)  # Add to the result list
                    except json.JSONDecodeError:
                        print(f"Unable to parse object: {obj_str}")
                    start_index = i + 1  # Update start index for the next object

    # Return the parsed objects list which is a valid JSON format
    return objects

# Example: Parsing a broken JSON file

def date_to_second(exp):
    t = exp.split("_")[-5:]
    # Compute a "timestamp" by converting parts into seconds
    sign = int(t[-1]) + int(t[-2]) * 60 + int(t[-3]) * 3600 + int(t[-4]) * 3600 * 24 + int(t[-5]) * 3600 * 24 * 50
    # Here the month duration is approximated as 50 days for simplicity in comparisons
    return sign


if os.path.exists("common_route_scores.json"):
    with open("common_route_scores.json", "r") as f:
        round_score_list, round_score_list_nofb = json.load(f)
else:        
    root_dir = "/GPFS/rhome/zijunwang/WorkSpace/V2Xverse/results/results_driving_0307_group_new"
    root_dir_nofb = "/GPFS/rhome/zijunwang/WorkSpace/V2Xverse/results/results_driving_0307_group_nofb"     
    round_score_list = [[] for _ in range(5)]
    round_score_list_nofb = [[] for _ in range(5)]
    v2x_final_path = Path(root_dir) / "image/v2x_final"
    v2x_final_nofb_path = Path(root_dir_nofb) / "image/v2x_final"
    for route in os.listdir(v2x_final_path):
        if not os.path.exists(v2x_final_nofb_path / route):
            continue    
        try:
            exp_name_list = os.listdir(v2x_final_path / route)
            latest_exp = sorted(exp_name_list, key=date_to_second)[-1]
            json_path = v2x_final_path / route / latest_exp / 'log/group_negotiation_records.json'
            records = parse_broken_json(json_path)
            for record in records:
                assert len(record) == 1
                for step, step_record in record.items():
                    if len(step_record) != 5:
                        continue
                    for r, round_record in step_record.items():
                        cons_score = round_record["scores"]["cons_score"]
                        eff_score = round_record["scores"]["efficiency_score"]
                        safety_score = round_record["scores"]["safety_score"]
                        # total_score = 3 * eff_score + 7 * safety_score
                        round_score_list[int(r)].append([cons_score, eff_score, safety_score])       
            exp_name_list = os.listdir(v2x_final_nofb_path / route)
            latest_exp = sorted(exp_name_list, key=date_to_second)[-1]
            json_path = v2x_final_nofb_path / route / latest_exp / 'log/group_negotiation_records.json'
            records = parse_broken_json(json_path)
            for record in records:
                assert len(record) == 1
                for step, step_record in record.items():
                    if len(step_record) != 5:
                        continue
                    for r, round_record in step_record.items():
                        cons_score = round_record["scores"]["cons_score"]
                        eff_score = round_record["scores"]["efficiency_score"]
                        safety_score = round_record["scores"]["safety_score"]
                        # total_score = 3 * eff_score + 7 * safety_score
                        round_score_list_nofb[int(r)].append([cons_score, eff_score, safety_score])
        except Exception as e:
            print(e)
    with open("common_route_scores.json", "w") as f:
        json.dump([round_score_list, round_score_list_nofb], f)
         
round_score_list = np.array(round_score_list)
round_score_list_nofb = np.array(round_score_list_nofb)
print(round_score_list.shape)
print(round_score_list_nofb.shape)
# import ipdb; ipdb.set_trace()
a = 1
b = 3
c = 6
a = (10 / (a+b+c))*a
b = (10 / (a+b+c))*b
c = (10 / (a+b+c))*c
round_score_list = round_score_list[:, :, 0] * a + round_score_list[:, :, 1] * b + round_score_list[:, :, 2] * c
round_score_list_nofb = round_score_list_nofb[:, :, 0] * a + round_score_list_nofb[:, :, 1] * b + round_score_list_nofb[:, :, 2] * c

y_max = 10
y_min = 6
y_lim = 0.5
x = [1, 2, 3, 4, 5]
colors = ['b', 'g', 'r', 'y', 'm']
plt.rcParams.update({'font.size': 24})
for r, round_scores in enumerate(round_score_list):
    if r == 3:
        break
    print(r, len(round_scores))
    kde = gaussian_kde(round_scores)
    y_values = np.linspace(y_min, y_max, 1000)  # Define the range of y values
    pdf = kde(y_values)
    plt.plot(y_values, pdf, label=f'round {r + 1}', color=colors[r])
    
# Set chart title and labels
plt.title('Negotiation w/ feedback')
plt.xlabel('Scores')
plt.ylabel('Probability Density')
plt.ylim(0, y_lim)
plt.legend(fontsize=18)
# Save the chart 
plt.tight_layout()
plt.savefig('vis_results/common_total_score_distribution.png')
plt.close()

for r, round_scores in enumerate(round_score_list_nofb):
    if r == 3:
        break
    print(r, len(round_scores))
    kde = gaussian_kde(round_scores)
    y_values = np.linspace(y_min, y_max, 1000)  # Define the range of y values
    pdf = kde(y_values)
    plt.plot(y_values, pdf, label=f'round {r + 1}', color=colors[r], linestyle='--')    

# Set chart title and labels
plt.title('Negotiation w/o feedback')
plt.xlabel('Scores')
plt.ylabel('Probability Density')
plt.ylim(0, y_lim)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('vis_results/common_total_score_distribution_nofb.png')
plt.close()

img1 = Image.open('vis_results/common_total_score_distribution.png')
img2 = Image.open('vis_results/common_total_score_distribution_nofb.png')
# Create a new image with width equal to the sum of both images and height equal to one image's height
new_img = Image.new('RGB', (img1.width + img2.width, img1.height))
new_img.paste(img1, (0, 0))
new_img.paste(img2, (img1.width, 0))
# Show or save the combined image
# new_img.show()
new_img.save('vis_results/common_total_score_distribution_compare.png')
# new_img.save("vis_results/cons_score.png")
