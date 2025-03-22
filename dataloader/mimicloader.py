from PIL import Image
import csv
import torch
from torchvision import transforms
import numpy as np
import pandas as pd

class MimicDataset(torch.utils.data.Dataset):
    def __init__(self, directory, image_size, is_test):
        super().__init__()
        self.image_size = image_size
        self.directory = directory
        self.is_test = is_test

        if is_test == False:
            self.label_csv_dir = self.directory + '/preprocessed_mimic/mimic_train_version_one_five.csv'
        else:
            self.label_csv_dir = self.directory + '/preprocessed_mimic/mimic_test_version_one_five.csv'

        self.image_paths = []
        self.demographics = []
        self.ehrs = []
        self.labels = []
        self.is_demo_categorical = [1, 0, 1] # [sex, age, ap/pa]
        self.is_ehr_categorical = [0, 0, 0, 0, 0] # [BMI (kg/m2),Blood Pressure Max,Blood Pressure Min,Height (Inches),Weight (Lbs)]

        with open(self.label_csv_dir, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)
            label_index = header.index('No Finding')
            path_index = header.index('Path')
            sex_index = header.index('sex')
            age_index = header.index('age')
            appa_index = header.index('ViewPosition')

            bmi_index = header.index('BMI (kg/m2)')
            bp_max_index = header.index('Blood Pressure Max')
            bp_min_index = header.index('Blood Pressure Min')
            height_index = header.index('Height (Inches)')
            weight_index = header.index('Weight (Lbs)')

            for i, row in enumerate(reader):
                
                if is_test == False:
                    if row[label_index] == '1.0':
                        demo = [0, 0, 0]
                        if row[sex_index] == '1.0':
                            demo[0] = 1
                        elif row[sex_index] == '0.0':
                            demo[0] = 0
                        else:
                            exit('sex label error ...')
                        
                        demo[1] = float(row[age_index])

                        if row[appa_index] == 'AP':
                            demo[2] = 1
                        elif row[appa_index] == 'PA':
                            demo[2] = 0
                        else:
                            exit('appa label error ...', row[appa_index])
                        
                        self.demographics.append(demo)

                        ehr = [0, 0, 0, 0, 0]
                        ehr[0] = float(row[bmi_index])
                        ehr[1] = float(row[bp_max_index])
                        ehr[2] = float(row[bp_min_index])
                        ehr[3] = float(row[height_index])
                        ehr[4] = float(row[weight_index])

                        self.ehrs.append(ehr)

                        path = self.directory + '/mimic-cxr-jpg/' + row[path_index]
                        self.image_paths.append(path)
                else:
                    if row[label_index] == '1.0': #No Finding => 1
                        self.labels.append([0])
                    else:
                        self.labels.append([1])

                    demo = [0, 0, 0]
                    if row[sex_index] == '1.0':
                        demo[0] = 1
                    elif row[sex_index] == '0.0':
                        demo[0] = 0
                    else:
                        exit('sex label error ...')
                    
                    demo[1] = float(row[age_index])

                    if row[appa_index] == 'AP':
                        demo[2] = 1
                    elif row[appa_index] == 'PA':
                        demo[2] = 0
                    else:
                        exit('appa label error ...', row[appa_index])
                    
                    self.demographics.append(demo)

                    ehr = [0, 0, 0, 0, 0]
                    ehr[0] = float(row[bmi_index])
                    ehr[1] = float(row[bp_max_index])
                    ehr[2] = float(row[bp_min_index])
                    ehr[3] = float(row[height_index])
                    ehr[4] = float(row[weight_index])

                    self.ehrs.append(ehr)

                    path = self.directory + '/mimic-cxr-jpg/' + row[path_index]
                    self.image_paths.append(path)
                
        self.demographics = np.array(self.demographics)
        ages = self.demographics[:, 1]

        min_age = 1
        max_age = 100
        assert ages.max() < max_age

        self.demographics[:, 1] = (ages - min_age) / (max_age - min_age)

        self.ehrs = np.array(self.ehrs)

        mean = np.mean(self.ehrs, axis=0)
        std = np.std(self.ehrs, axis=0)

        self.ehrs = (self.ehrs - mean) / std

    def __getitem__(self, index):
        # return 0

        image = Image.open(self.image_paths[index])
        # print(image.mode)

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        image_tensor = transform(image)
        # print(image_tensor.shape) #torch.Size([1, 256, 256])
    
        demo = self.demographics[index]
        demo_tensor = torch.FloatTensor(demo)

        is_demo_cat = torch.FloatTensor(self.is_demo_categorical)
        
        demo_indices = torch.arange(0, 5) 

        ehr = self.ehrs[index]
        ehr_tensor = torch.FloatTensor(ehr)

        is_ehr_cat = torch.FloatTensor(self.is_ehr_categorical)

        ehr_indices = torch.arange(0, 5)

        if self.is_test == False:
            return image_tensor, demo_tensor, is_demo_cat, demo_indices, ehr_tensor, is_ehr_cat, ehr_indices
        else:
            label = self.labels[index]
            label = torch.FloatTensor(label)
            return image_tensor, demo_tensor, is_demo_cat, demo_indices, ehr_tensor, is_ehr_cat, ehr_indices, label, self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)