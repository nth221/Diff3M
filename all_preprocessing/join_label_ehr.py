import csv
import numpy as np

def join_with_patients_using_subject_id(train_path, test_path, patients_path, result_path):
    patients_rows = []
    patients_header = []
    with open(patients_path, mode='r') as file:
        reader = csv.reader(file)
        list_reader = list(reader)

        patinets_header = list_reader[0]
        patients_rows = list_reader[1:]     
        # patients_header = next(reader)
    # print(patients_rows[:5][:, 0])

    patients_rows = np.array(patients_rows)
    # print(patients_rows[:5, 0])

    start_index = 0
    train_header = []
    patient_temp = ''
    with open(train_path, mode='r') as file:
        reader = csv.reader(file)
        train_header = next(reader)

        train_header.extend(['sex', 'age'])

        with open(result_path + '/mimic_first_joined_train.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(train_header)

        for idx, row in enumerate(reader):
            path = row[0]
            patient_id = path.split('/')[3]
            assert 'p' in patient_id
            
            patient_id = patient_id[1:]

            if idx == 0:
                patient_temp = patient_id                


            found_id_array = np.where(patients_rows[start_index:, 0]==patient_id)[0]
            
            if found_id_array.size > 0:
                found_row_id = found_id_array[0]
            else:
                continue

            sex_str = patients_rows[start_index + found_row_id, 1]
            age_str = patients_rows[start_index + found_row_id, 2]

            if patient_temp != patient_id:
                start_index = found_row_id
                patient_temp = patient_id

            sex_float = 0
            if sex_str == 'M':
                sex_float = 1.0
            elif sex_str == 'F':
                sex_float = 0.0
            else:
                exit('sex error ...')
            
            age_float = float(age_str)

            row.append(sex_float)
            row.append(age_float)

            print('train-',idx, row)
            with open(result_path + '/mimic_first_joined_train.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)

        
    start_index = 0
    test_header = []
    patient_temp = ''
    with open(test_path, mode='r') as file:
        reader = csv.reader(file)
        test_header = next(reader)

        test_header.extend(['sex', 'age'])

        with open(result_path + '/mimic_first_joined_test.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(test_header)

        for idx, row in enumerate(reader):
            path = row[0]
            patient_id = path.split('/')[3]
            assert 'p' in patient_id
            
            patient_id = patient_id[1:]

            if idx == 0:
                patient_temp = patient_id                


            found_id_array = np.where(patients_rows[start_index:, 0]==patient_id)[0]
            
            if found_id_array.size > 0:
                found_row_id = found_id_array[0]
            else:
                continue

            sex_str = patients_rows[start_index + found_row_id, 1]
            age_str = patients_rows[start_index + found_row_id, 2]

            if patient_temp != patient_id:
                start_index = found_row_id
                patient_temp = patient_id


            sex_float = 0
            if sex_str == 'M':
                sex_float = 1.0
            elif sex_str == 'F':
                sex_float = 0.0
            else:
                exit('sex error ...')
            
            age_float = float(age_str)

            row.append(sex_float)
            row.append(age_float)

            print('test-', idx, row)
            with open(result_path + '/mimic_first_joined_test.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)        
    return

def join_with_mimic_plus_using_dicom_id(train_path, test_path, plus_path, result_path):
    ehr_rows = []
    with open(plus_path, mode='r') as file:
        reader = csv.reader(file)
        list_reader = list(reader)

        header = list_reader[0]
        ehr_rows = list_reader[1:]
    
    ehr_rows = np.array(ehr_rows)

    start_index = 0
    train_header = []
    with open(train_path, mode='r') as file:
        reader = csv.reader(file)
        train_header = next(reader)

        train_header.extend(['BMI (kg/m2)','Blood Pressure Max','Blood Pressure Min','Height (Inches)','Weight (Lbs)'])

        with open(result_path + '/mimic_second_joined_train.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(train_header)
        
        for idx, row in enumerate(reader):
            path = row[0]
            dicom_id = path.split('/')[5]

            assert '.jpg' in dicom_id

            dicom_id = dicom_id[:-4]
            
            found_id_array = np.where(ehr_rows[start_index:, 0]==dicom_id)[0]

            if found_id_array.size > 0:
                found_row_id = found_id_array[0]
            else:
                continue
                
            ehrs = ehr_rows[start_index + found_row_id, 3:]

            start_index = found_row_id

            # print(dicom_id, found_row_id)

            row.extend(ehrs)
            print('train-', idx, row)
            with open(result_path + '/mimic_second_joined_train.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            


    start_index = 0
    test_header = []
    with open(test_path, mode='r') as file:
        reader = csv.reader(file)
        test_header = next(reader)

        test_header.extend(['BMI (kg/m2)','Blood Pressure Max','Blood Pressure Min','Height (Inches)','Weight (Lbs)'])

        with open(result_path + '/mimic_second_joined_test.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(test_header)
        
        for idx, row in enumerate(reader):
            path = row[0]
            dicom_id = path.split('/')[5]

            assert '.jpg' in dicom_id

            dicom_id = dicom_id[:-4]
            
            found_id_array = np.where(ehr_rows[start_index:, 0]==dicom_id)[0]

            if found_id_array.size > 0:
                found_row_id = found_id_array[0]
            else:
                continue
                
            ehrs = ehr_rows[start_index + found_row_id, 3:]

            start_index = found_row_id


            row.extend(ehrs)
            print('test-', idx, row)
            with open(result_path + '/mimic_second_joined_test.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)

    return

def main():
    mimic_cxr_train_path = '/data2/hkim/datasets/mimic-cxr-jpg/2.0.0/train.csv'
    mimic_cxr_test_path = '/data2/hkim/datasets/mimic-cxr-jpg/2.0.0/test.csv' 
    
    mimic_second_train_path = '/data2/hkim/datasets/preprocessed_mimic/mimic_first_joined_train.csv'
    mimic_second_test_path = '/data2/hkim/datasets/preprocessed_mimic/mimic_first_joined_test.csv'

    mimic_iv_patients_path = '/data2/hkim/datasets/mimiciv/3.1/hosp/patients.csv'
    mimic_plus_path = '/data2/hkim/datasets/mimic+/concated_mimic_plus.csv'

    result_path = '/data2/hkim/datasets/preprocessed_mimic'
    
    join_with_patients = False
    join_with_mimic_plus = True

    if join_with_patients == True:
        join_with_patients_using_subject_id(
            mimic_cxr_train_path,
            mimic_cxr_test_path,
            mimic_iv_patients_path,
            result_path
        )
    if join_with_mimic_plus == True:
        join_with_mimic_plus_using_dicom_id(
            mimic_second_train_path,
            mimic_second_test_path,        
            mimic_plus_path,
            result_path
        )


    return

if __name__ == "__main__":
    main()