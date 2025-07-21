import json
# import pdb
import os

def check_log_file(file_path, target_string):
    try:
        # 打开文件
        with open(file_path, 'r') as file:
            # 读取文件内容
            file_content = file.read()

            # 判断文件内容中是否包含特定字符
            if target_string in file_content:
                # print(f"文件中包含目标字符串 '{target_string}'")
                return True
            else:
                # print(f"文件中不包含目标字符串 '{target_string}'")
                return False

    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在")

if __name__ == '__main__':
    root_path = "/GPFS/data/changxingliu/V2Xverse/results/results_driving_re_colmdriver_multiple_realtime_interrupt_05_0512"
    title = root_path.split('/')[-1].split('_')[-1]

    route_id_bank_list = ["r1_town05_ins_c", "r2_town05_ins_c", "r3_town05_ins_c", "r4_town06_ins_c", "r5_town06_ins_c", "r6_town07_ins_c", "r7_town05_ins_ss", "r8_town05_ins_ss", "r9_town06_ins_ss", "r10_town07_ins_ss", "r11_town05_ins_sl", "r12_town06_ins_sl", "r13_town05_ins_sl", "r14_town07_ins_sl", "r15_town07_ins_sl", "r16_town05_ins_sl", "r17_town05_ins_sr", "r18_town05_ins_sr", "r19_town05_ins_sr", "r20_town06_ins_sr", "r21_town07_ins_sr", "r22_town07_ins_sr", "r23_town05_ins_oppo", "r24_town05_ins_rl", "r25_town05_ins_crosschange", "r26_town05_ins_chaos", "r27_town06_hw_merge", "r28_town06_hw_c", "r29_town06_hw_merge", "r30_town06_hw_merge", "r31_town05_ins_oppo", "r32_town05_ins_oppo", "r33_town05_ins_rl", "r34_town05_ins_rl", "r35_town05_ins_crosschange", "r36_town05_ins_crosschange", "r37_town05_ins_chaos", "r38_town05_ins_chaos", "r39_town06_hw_c", "r40_town06_hw_c", "r41_town05_ins_oppo", "r42_town05_ins_rl", "r43_town05_ins_crosschange", "r44_town05_ins_chaos", "r45_town06_hw_merge", "r46_town06_hw_c"]
    
    excel_file = 'val_results_coop.xlsx'
    # for i in range(len(route_id_bank_list)):
    #     route_id_bank_list.append(route_id_bank_list[i]+'_1')
    #     route_id_bank_list[i] = route_id_bank_list[i]+'_0'
    # excel_file = 'val_results_single.xlsx'

    image_result_path = os.path.join(root_path, 'image/v2x_final')
    numerical_result_path = os.path.join(root_path, 'v2x_final')

    invalid_str_list = ['Traceback', 'traceback']

    save_as_excel = []

    for route in route_id_bank_list:
        reported_route_id = []

        if os.path.exists("{}/{}/ego_vehicle_2/results.json".format(numerical_result_path, route)):
            mode = 'triple'
            file_path_0 = "{}/{}/ego_vehicle_0/results.json".format(numerical_result_path, route)
            file_path_1 = "{}/{}/ego_vehicle_1/results.json".format(numerical_result_path, route)
            file_path_2 = "{}/{}/ego_vehicle_2/results.json".format(numerical_result_path, route)
        elif os.path.exists("{}/{}/ego_vehicle_1/results.json".format(numerical_result_path, route)):
            mode = 'double'
            file_path_0 = "{}/{}/ego_vehicle_0/results.json".format(numerical_result_path, route)
            file_path_1 = "{}/{}/ego_vehicle_1/results.json".format(numerical_result_path, route)
        else:
            mode = 'single'
            file_path = "{}/{}/ego_vehicle_0/results.json".format(numerical_result_path, route)
        try :
            log_file_path = "{}/{}/log/log.log".format(numerical_result_path, route)
            if not os.path.exists(log_file_path):
                log_list = os.listdir("{}/{}".format(image_result_path, route))
                log_list.sort()
                log_dir = log_list[-1]
                log_file_path = "{}/{}/{}/log/log.log".format(image_result_path, route, log_dir)
            valid_flag = True
            invalid_type = ''

            for target_string in invalid_str_list:
                check_result = check_log_file(log_file_path, target_string)
                valid_flag *= not check_result
                if check_result:
                    invalid_type += target_string[:6]

            if mode=='single':
                with open(file_path, 'r') as fcc_file:
                    fcc_data = json.load(fcc_file)

                records = fcc_data["_checkpoint"]["records"]
            elif mode=='double':
                with open(file_path_0, 'r') as fcc_file_0, open(file_path_1, 'r') as fcc_file_1:
                    fcc_data_0 = json.load(fcc_file_0)
                    fcc_data_1 = json.load(fcc_file_1)

                records = []
                original_record_0 = fcc_data_0["_checkpoint"]["records"]
                original_record_1 = fcc_data_1["_checkpoint"]["records"]
                records = original_record_0
                for i in range(len(original_record_0)):
                    records[i]['scores']['score_route'] = (original_record_0[i]['scores']['score_route'] + original_record_1[i]['scores']['score_route']) / 2
                    records[i]['scores']['score_penalty'] = (original_record_0[i]['scores']['score_penalty'] + original_record_1[i]['scores']['score_penalty']) / 2
                    records[i]['scores']['score_composed'] = (original_record_0[i]['scores']['score_composed'] + original_record_1[i]['scores']['score_composed']) / 2
            elif mode=='triple':
                with open(file_path_0, 'r') as fcc_file_0, open(file_path_1, 'r') as fcc_file_1, open(file_path_2, 'r') as fcc_file_2:
                    fcc_data_0 = json.load(fcc_file_0)
                    fcc_data_1 = json.load(fcc_file_1)
                    fcc_data_2 = json.load(fcc_file_2)

                records = []
                original_record_0 = fcc_data_0["_checkpoint"]["records"]
                original_record_1 = fcc_data_1["_checkpoint"]["records"]
                original_record_2 = fcc_data_2["_checkpoint"]["records"]
                records = original_record_0
                for i in range(len(original_record_0)):
                    records[i]['scores']['score_route'] = (original_record_0[i]['scores']['score_route'] + original_record_1[i]['scores']['score_route'] + original_record_2[i]['scores']['score_route']) / 3
                    records[i]['scores']['score_penalty'] = (original_record_0[i]['scores']['score_penalty'] + original_record_1[i]['scores']['score_penalty'] + original_record_2[i]['scores']['score_penalty']) / 3
                    records[i]['scores']['score_composed'] = (original_record_0[i]['scores']['score_composed'] + original_record_1[i]['scores']['score_composed'] + original_record_2[i]['scores']['score_composed']) / 3
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

                print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\t{}'.format(
                    title+'_'+route, record['scores']['score_route'], record['scores']['score_penalty']/red_light_penalty, record['scores']['score_composed']/red_light_penalty,
                    record['meta']['duration_game'], record['meta']['duration_system'],
                    int(valid_flag),
                    invalid_type
                    ))
                reported_route_id.append(route)

                save_as_excel.append([
                    title+'_'+route, record['scores']['score_route'], record['scores']['score_penalty']/red_light_penalty, record['scores']['score_composed']/red_light_penalty,
                    record['meta']['duration_game'], record['meta']['duration_system'],
                    int(valid_flag),
                    invalid_type
                    ])
                    
        except:
            print(title+'_'+route)

    len_excel = len(save_as_excel)
    print(len_excel)
    print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(
        'average', sum([a[1] for a in save_as_excel])/len_excel, sum([a[2] for a in save_as_excel])/len_excel, sum([a[3] for a in save_as_excel])/len_excel,
        sum([a[4] for a in save_as_excel])/len_excel, sum([a[5] for a in save_as_excel])/len_excel))

    from pandas import DataFrame
    df = DataFrame({'name': [a[0] for a in save_as_excel], 'score_route': [a[1] for a in save_as_excel], 
                    'score_penalty': [a[2] for a in save_as_excel], 'score_composed': [a[3] for a in save_as_excel],
                    'duration_game': [a[4] for a in save_as_excel], 'duration_system': [a[5] for a in save_as_excel],
                    'valid_flag': [a[6] for a in save_as_excel], 'invalid_type': [a[7] for a in save_as_excel]})
    df.to_excel(os.path.join(root_path, excel_file), sheet_name='sheet1', index=False)