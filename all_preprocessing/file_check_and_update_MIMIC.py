import csv
import os

def file_absence_check_update(train_path, test_path, result_path, data_root):
    train_eliminated = 0
    with open(train_path, mode='r') as file:
        reader = csv.reader(file)
        train_header = next(reader)

        with open(result_path + '/mimic_third_train.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(train_header)
        
        for idx, row in enumerate(reader):
            path = row[0]

            if os.path.exists(data_root + '/' + path):
                with open(result_path + '/mimic_third_train.csv', mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
            else:
                train_eliminated += 1

    test_eliminated = 0
    with open(test_path, mode='r') as file:
        reader = csv.reader(file)
        test_header = next(reader)

        with open(result_path + '/mimic_third_test.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(test_header)
        
        for idx, row in enumerate(reader):
            path = row[0]

            if os.path.exists(data_root + '/' + path):
                with open(result_path + '/mimic_third_test.csv', mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
            else:
                test_eliminated += 1
    
    print('train eliminated : ', train_eliminated)
    print('test eliminated : ', test_eliminated)
    return 0

def main():
    mimic_second_train_path = '/data3/hkim/datasets/preprocessed_mimic/mimic_second_joined_train.csv'
    mimic_second_test_path = '/data3/hkim/datasets/preprocessed_mimic/mimic_second_joined_test.csv'

    result_path = '/data3/hkim/datasets/preprocessed_mimic'
    
    data_root = '/data3/hkim/datasets/mimic-cxr-jpg'

    file_absence_check_update(mimic_second_train_path, mimic_second_test_path, result_path, data_root)


if __name__ == "__main__":
    main()