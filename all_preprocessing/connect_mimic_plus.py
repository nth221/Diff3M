import os
import csv
import glob as glob

def main():
    mimic_plus_path = '/data2/hkim/datasets/mimic+'

    csv_list = glob.glob(mimic_plus_path + '/*.csv')    
    csv_list.sort()

    # print(csv_list)
    all_dicoms = []
    for idx, csv_file in enumerate(csv_list):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            list_reader = list(reader)

            if idx == 0:
                all_dicoms.append(list_reader[0])
            
            all_dicoms.extend(list_reader[1:])
    

    print(len(all_dicoms))
    with open(mimic_plus_path + '/concated_mimic_plus.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(all_dicoms)

    return

if __name__ == "__main__":
    main()