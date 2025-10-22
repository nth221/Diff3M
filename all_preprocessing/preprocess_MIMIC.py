
import csv
import copy
from datetime import datetime
import multiprocessing

result_save_root = '/data2/hkim/datasets/mimic+'

def match_cxr_study_id_with_omr_results(metadata_path, omr_path, max_diff_day, remove_result_name, start_idx, end_idx, process_num):
    dicom_ids = []
    subject_ids = []
    study_dates = [] #7
    with open(metadata_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)

        assert header[0] == 'dicom_id'
        assert header[1] == 'subject_id'
        assert header[7] == 'StudyDate'

        for i, row in enumerate(reader):
            if start_idx <= i and i < end_idx:
                dicom_ids.append(row[0])
                subject_ids.append(row[1])
                study_dates.append(row[7])
            elif i >= end_idx:
                break
    
    result_name = []
    with open(omr_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for i, row in enumerate(reader):
            result_name.append(row[3])
    
    result_name_set = list(set(result_name))
    result_name_set.sort() #'BMI', 'BMI (kg/m2)', 'Blood Pressure', 'Blood Pressure Lying', 'Blood Pressure Sitting', 'Blood Pressure Standing', 'Blood Pressure Standing (1 min)', 'Blood Pressure Standing (3 mins)', 'Height', 'Height (Inches)', 'Weight', 'Weight (Lbs)', 'eGFR'
    
    for rm_name in remove_result_name:
        result_name_set.remove(rm_name)

    result_name_set.remove('BMI')
    result_name_set.remove('Height')
    result_name_set.remove('Weight')

    temp_header = []
    for rn in result_name_set:
        if 'Pressure' in rn:
            temp_header.append(rn + ' Max')
            temp_header.append(rn + ' Min')
        else:
            temp_header.append(rn)

    # print(result_name_set)
    result_header = ['dicom_id', 'subject_id', 'StudyDate']
    result_header.extend(temp_header)

    print('new table header:', result_header) # ['dicom_id', 'subject_id', 'StudyDate', 'BMI (kg/m2)', 'Blood Pressure Max', 'Blood Pressure Min', 'Height (Inches)', 'Weight (Lbs)']

    # result_rows = []

    with open(result_save_root + '/mimic_cxr+ehr_start-' + str(start_idx) + '_end-' + str(end_idx) + '.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(result_header)
    # result_rows.append(result_header)

    dup_check_subject_id = ''
    dup_check_study_date = ''
    progress_total = len(dicom_ids)
    for idx, dicom_id in enumerate(dicom_ids):
        print('Process #'+str(process_num) + '--' + str(int((idx / progress_total) * 100)) + '%')
        subject_id = subject_ids[idx]
        study_date = datetime.strptime(study_dates[idx], '%Y%m%d').date()

        if dup_check_subject_id == subject_id and dup_check_study_date == study_date:
            result_row[0] = dicom_id
            with open(result_save_root + '/mimic_cxr+ehr_start-' + str(start_idx) + '_end-' + str(end_idx) + '.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(result_row)
            # result_rows.append(result_row)
            # print(idx, result_row) 
            continue
        else:
            dup_check_subject_id = subject_id
            dup_check_study_date = study_date

        result_row = [0] * len(result_header)

        result_row[0] = dicom_id
        result_row[1] = subject_id
        result_row[2] = study_dates[idx]

        for rdx, result_name in enumerate(result_name_set):
            temp_omr_results = []

            subject_id_start = False
            with open(omr_path, mode='r') as file:

                reader = csv.reader(file)
                header = next(reader) #subject_id,chartdate,seq_num,result_name,result_value
                # print(header[0])
                # print(header[3])

                for jdx, row in enumerate(reader):                        
                    if row[0] == subject_id and row[3] in result_name:
                        if subject_id_start == False:
                            subject_id_start = True                            
                        omr_date = datetime.strptime(row[1], '%Y-%m-%d').date()
                        # print('date1', study_date, 'date2', omr_date)
                        day_diff = abs((study_date - omr_date).days)
                        # print('diff', day_diff)

                        if day_diff <= max_diff_day:
                            temp_omr_results.append(row)
                    if row[0] != subject_id and subject_id_start == True:
                        break                        
                    # print(row[0], row[3])
                    # print(subject_id, result_name)
            if len(temp_omr_results) == 0:
                continue
            elif len(temp_omr_results) == 1:
                if 'Pressure' in result_name:
                    result_row[result_header.index(result_name + ' Max')] = temp_omr_results[0][4].split('/')[0]
                    result_row[result_header.index(result_name + ' Min')] = temp_omr_results[0][4].split('/')[-1]
                else:
                    result_row[result_header.index(result_name)] = temp_omr_results[0][4]
            else:
                min_day_diff = 99999
                min_date = ''
                for temp_omr in temp_omr_results:
                    temp_omr_date = datetime.strptime(temp_omr[1], '%Y-%m-%d').date()
                    temp_day_diff = abs((study_date - temp_omr_date).days)
                    if temp_day_diff < min_day_diff:
                        min_date = temp_omr[1]
                        min_day_diff = temp_day_diff
                    if temp_omr[1] != min_date and temp_day_diff == min_day_diff:
                        min_date = min_date + '/' + temp_omr[1]


                
                temp_max_values = 0
                temp_min_values = 0
                count = 0
                for temp_omr in temp_omr_results:
                    if temp_omr[1] in min_date:
                        if 'Pressure' in result_name:
                            temp_max_values += float(temp_omr[4].split('/')[0])
                            temp_min_values += float(temp_omr[4].split('/')[-1])
                        else:
                            temp_max_values += float(temp_omr[4])
                        count += 1
                if 'Pressure' in result_name:
                    result_row[result_header.index(result_name + ' Max')] = str(temp_max_values / count)
                    result_row[result_header.index(result_name + ' Min')] = str(temp_min_values / count)
                else:
                    result_row[result_header.index(result_name)] = str(temp_max_values / count)

        with open(result_save_root + '/mimic_cxr+ehr_start-' + str(start_idx) + '_end-' + str(end_idx) + '.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(result_row)
        # result_rows.append(result_row)
        # print(idx, result_row) 


    # assert len(dicom_ids) == len(set(dicom_ids))



def main():
    mimic_cxr_metadata_path = '/data2/hkim/datasets/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
    mimic_iv_omr_path = '/data2/hkim/datasets/mimiciv/3.1/hosp/omr.csv'

    max_diff_day = 90

    remove_result_name = ['Blood Pressure Lying','Blood Pressure Sitting', 'Blood Pressure Standing', 'Blood Pressure Standing (1 min)', 'Blood Pressure Standing (3 mins)', 'eGFR']

    total_row_number = 377110 # header 제외
    rows_per_process = 15000

    process_num = total_row_number // rows_per_process + 1
    
    processes = []
    for idx in range(process_num - 1):
        p = multiprocessing.Process(target=match_cxr_study_id_with_omr_results, args=(mimic_cxr_metadata_path, mimic_iv_omr_path, max_diff_day, remove_result_name, idx * rows_per_process, (idx + 1) * rows_per_process, idx))
        processes.append(p)
        p.start()
    p = multiprocessing.Process(target=match_cxr_study_id_with_omr_results, args=(mimic_cxr_metadata_path, mimic_iv_omr_path, max_diff_day, remove_result_name, (process_num - 1) * rows_per_process, total_row_number, process_num -1))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()