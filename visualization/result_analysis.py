import json
import sys
import os
import pandas as pd

def check_log_file(file_path, target_string):
    try:
        # Open file
        with open(file_path, 'r') as file:
            # Read file content
            file_content = file.read()

            # Check if the file content contains the target string
            if target_string in file_content:
                return True
            else:
                return False

    except FileNotFoundError:
        print(f"'{file_path}' not found")

if __name__ == '__main__':
    args = sys.argv
    root_path = args[1] # 'results/results_driving_colmdriver'
    title = root_path.split('/')[-1].split('_')[-1]

    route_id_list = ["r1_town05_ins_c", "r2_town05_ins_c", "r3_town05_ins_c", "r4_town06_ins_c", "r5_town06_ins_c", "r6_town07_ins_c", "r7_town05_ins_ss", "r8_town05_ins_ss", "r9_town06_ins_ss", "r10_town07_ins_ss", "r11_town05_ins_sl", "r12_town06_ins_sl", "r13_town05_ins_sl", "r14_town07_ins_sl", "r15_town07_ins_sl", "r16_town05_ins_sl", "r17_town05_ins_sr", "r18_town05_ins_sr", "r19_town05_ins_sr", "r20_town06_ins_sr", "r21_town07_ins_sr", "r22_town07_ins_sr", "r23_town05_ins_oppo", "r24_town05_ins_rl", "r25_town05_ins_crosschange", "r26_town05_ins_chaos", "r27_town06_hw_merge", "r28_town06_hw_c", "r29_town06_hw_merge", "r30_town06_hw_merge", "r31_town05_ins_oppo", "r32_town05_ins_oppo", "r33_town05_ins_rl", "r34_town05_ins_rl", "r35_town05_ins_crosschange", "r36_town05_ins_crosschange", "r37_town05_ins_chaos", "r38_town05_ins_chaos", "r39_town06_hw_c", "r40_town06_hw_c", "r41_town05_ins_oppo", "r42_town05_ins_rl", "r43_town05_ins_crosschange", "r44_town05_ins_chaos", "r45_town06_hw_merge", "r46_town06_hw_c"]
    route_id_bank_list = [prefix + route_id for route_id in route_id_list for prefix in ['', 'Interdrive_', 'Interdrive_npc_', 'Interdrive_no_npc_']]
    invalid_str_list = []
    save_as_excel = []
    routetype_dict = ["ins_c","ins_ss","ins_sr","ins_sl","ins_oppo","ins_rl","ins_crosschange","ins_chaos","hw_merge","hw_c"]

    analysis = {'score_composed':[0] * len(routetype_dict), 'score_route':[0] * len(routetype_dict), 'score_penalty':[0] * len(routetype_dict), 'success_rate':[0] * len(routetype_dict), 'game_time':[0] * len(routetype_dict)}
    cnt = [0] * len(routetype_dict)

    for route in route_id_bank_list:
        reported_route_id = []
        file_path = []
        if  os.path.isdir(os.path.join(root_path,'v2x_final')):
            route_dir = os.path.join(root_path,'v2x_final',route)
        else:
            route_dir = os.path.join(root_path,route)
        ego_num = 0

        try :
            for folder_name in os.listdir(route_dir):
                folder_path = os.path.join(route_dir, folder_name)
                if os.path.isdir(folder_path) and folder_name.startswith('ego_vehicle'):
                    ego_num += 1

            for i in range(ego_num):
                if os.path.isdir(os.path.join(root_path,'v2x_final')):
                    file_path.append(f"{root_path}/v2x_final/{route}/ego_vehicle_{i}/results.json")
                else:
                    file_path.append(f"{root_path}/{route}/ego_vehicle_{i}/results.json")
        
            valid_flag = True
            fcc_data = []
            original_record = []

            for i in range(ego_num):
                with open(file_path[i], 'r') as fcc_file:
                    fcc_data.append(json.load(fcc_file))
                original_record.append(fcc_data[i]["_checkpoint"]["records"])

            records = []
            records = original_record[0]

            for i in range(len(original_record[0])):
                records[i]['scores']['score_route'] = sum(original_record[j][i]['scores']['score_route'] for j in range(ego_num)) / ego_num
                records[i]['scores']['score_penalty'] = 1
                for j in range(ego_num):
                    records[i]['scores']['score_penalty'] *= original_record[j][i]['scores']['score_penalty']
                records[i]['scores']['score_composed'] = records[i]['scores']['score_route'] * records[i]['scores']['score_penalty']

            for record in records:
                red_light = len(record['infractions']['red_light'])
                stop_sign = len(record['infractions']['stop_infraction'])
                red_light_penalty = 1
                stop_sign_penalty = 1
                for i in range(red_light):
                    red_light_penalty *= 0.7
                for i in range(stop_sign):
                    stop_sign_penalty *= 0.8
                red_light_penalty *= stop_sign_penalty

                print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
                   route, record['scores']['score_route'], record['scores']['score_penalty']/red_light_penalty, record['scores']['score_composed']/red_light_penalty,
                   record['meta']['duration_game'], record['meta']['duration_system']
                   ))
                reported_route_id.append(route)

                save_as_excel.append([
                    route, record['scores']['score_route'], record['scores']['score_penalty']/red_light_penalty, record['scores']['score_composed']/red_light_penalty,
                    record['meta']['duration_game'], record['meta']['duration_system']
                    ])

                
                for i in range(len(routetype_dict)):
                    if routetype_dict[i] in route:
                        if i == 0 and ("crosschange" in route or "chaos" in route):
                            continue
                        analysis['score_route'][i] += record['scores']['score_route']
                        analysis['score_penalty'][i] += record['scores']['score_penalty']/red_light_penalty
                        analysis['score_composed'][i] += record['scores']['score_composed']/red_light_penalty
                        analysis['game_time'][i] += record['meta']['duration_game']
                        if record['scores']['score_composed']/red_light_penalty > 99.95:
                            analysis['success_rate'][i] += 1
                        cnt[i] += 1
                    
        except Exception as e:
            # raise
            print(route)

    val = list(analysis.values())
    
    val = [[x[i] / max(cnt[i], 1) for i in range(len(routetype_dict))] for x in val]
    val = list(map(list, zip(*val)))

    print(root_path,":multiple_scenarios")
    print(cnt)

    for i in range(len(routetype_dict)):
        print(routetype_dict[i].ljust(20), end='')
        for num in val[i]:
            if num>1:
                print(format(num, '.2f').ljust(6), end='\t')
            else:
                print(format(num, '.3f').ljust(6), end='\t')
        print('')
    print('')

    # Save results to Excel
    save_dir = root_path # f"results/stats/{args[1]}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame
    columns = ['score_composed', 'score_route', 'score_penalty', 'success_rate', 'game_time']
    df = pd.DataFrame(val, columns=columns, index=routetype_dict)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, "analysis_results.csv")
    df.to_csv(csv_path, index=True)
    print(f"Results saved to: {csv_path}")
