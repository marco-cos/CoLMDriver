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



# root_path = "/GPFS/data/gjliu-1/TPAMI/V2Xverse/results/results_driving_codriving/"  #single_final   /  collab
root_path = "/GPFS/data/changxingliu/V2Xverse/results/results_driving_internvl_8b_socket_waypoints_0726"  #single_final   /  collab
image_result_path = os.path.join(root_path, 'image/v2x_final/town05_short_collab')
numerical_result_path = os.path.join(root_path, 'v2x_final/town05_short_collab')

repeat = '0' 

# rout_id_bank_list = [[2,3,5,8,10,13,18,31,135,146,147,161,300,310,326,327],
#                 [2,3,7,12,19,22,24,25,27,136,138,161,300,320,321,322,328,329],
#                 [7,8,16,31,300,301,305,310,311,313,314,316,320,324,329,330,331],
#                 [1,8,9,12,14,15,16,28,140,145,160,311,312,317,321,323,328,331],
#                 [1,6,7,8,18,20,28,31,142,145,146,157,306,310,315,318],]
rout_id_bank_list = [[2,3,5,8,10,13,18,31,135,146,147,161,300,310,326,327]]

# rout_id_bank_list = [[0,1,5,136,306,307],
#                 [3,4,9, 20, 137, 139 ,140, 151, 154, 301, 308, 318, 327, 330],
#                 [1 ,6, 7, 136 ,137 ,138 ,141 ,150 ,151 ,154 ,155, 164, 300 ,305, 311 ,313, 315],
#                 [6 ,7,9, 10, 12,14, 15 ,16 ,23 ,147,157,160,307	,316,317,319,326,330],
#                 [1 ,2, 3 ,6 ,7 ,12 ,13,135 ,139	,140,160,163,311,312,319,326],]

# setting_list = ['0','0','0','0','0']

# setting_list = ['_1228',  '_1229', '_1230_2', '_0102', '_0104']
# setting_list = ['_1228_8_10',  '_1229_5_10', '_1230_2_5_10', '_0102_8_10', '_0104_5_50']
# setting_list = ['_1_8_10',  '_2_5_10', '_3_5_10', '_4_8_10', '_5_5_50']
setting_list = ['_1_8_10']

invalid_str_list = ['Traceback', 'traceback']#, 'sensor'  'memory', 'sensor took too long', 

for k, (setting_name, rout_id_bank) in enumerate(zip(setting_list,rout_id_bank_list)):
    reported_route_id = []

    # print('-----------repeat:', repeat)
    path_list = os.listdir(numerical_result_path)
    path_list.sort()
    setting_name_repeat = repeat + setting_name
    path_list_filte_repeat = [route for route in path_list if route.endswith(str(setting_name_repeat))]
    for route in path_list_filte_repeat:
        route_id = route[:-8][1:]
        route_id_number = route_id.split('_')[0]

        if not int(route_id_number) in rout_id_bank_list[k]:
            continue

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

            with open(file_path, 'r') as fcc_file:
                fcc_data = json.load(fcc_file)

            records = fcc_data["_checkpoint"]["records"]
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

                

                if True: #route.endswith('4'):
                    print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\t{}'.format(
                        int(route_id_number)+k*1000, record['scores']['score_route'], record['scores']['score_penalty']/red_light_penalty, record['scores']['score_composed']/red_light_penalty,
                        record['meta']['duration_game'], record['meta']['duration_system'],
                        int(valid_flag),
                        invalid_type
                        ))
                    reported_route_id.append(route_id_number)
                    
        except:
            # print('error')
            print(int(route_id_number)+k*1000)
            reported_route_id.append(route_id_number)
    for route_id_defaut in rout_id_bank:
        if not str(route_id_defaut) in reported_route_id:
            print(route_id_defaut+k*1000)
