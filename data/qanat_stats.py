import glob
import SimpleITK as sitk
import sys

module_root = '..'
sys.path.append(module_root)
import os
from settings import raw_folder, sheets_folder, ccf_folder
from utils.helpers import get_label_shape_stats
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

if __name__ == '__main__':
    root_folder = os.path.join(raw_folder, 'third_attempt')
    output_ccf_directory = os.path.join(ccf_folder, 'third_attempt')
    if not os.path.isdir(output_ccf_directory):
        os.mkdir(output_ccf_directory)
    d = []
    for data_source in ['train', 'test']:
        print('-' * 50)
        data_folder = os.path.join(root_folder, data_source)
        corona_folders = sorted([os.path.join(data_folder, folder) for folder in os.listdir(data_folder)])
        n_qanats = 0
        for corona_folder in corona_folders:
            files = sorted(glob.glob(corona_folder + '/*label*.nrrd'))
            label_files = [file for file in files if 'roi' not in file]
            # print('no of labels in {} is {}'.format(corona_folder, len(label_files)))
            for label_file in label_files:
                label = sitk.ReadImage(label_file)
                # ccf
                ccf = sitk.ConnectedComponentImageFilter()
                ccf.SetFullyConnected(True)
                ccf_labelmap = ccf.Execute(label)
                outpu_ccf_file_path = os.path.join(output_ccf_directory, os.path.basename(label_file))
                sitk.WriteImage(ccf_labelmap, outpu_ccf_file_path)
                stats_filter = sitk.LabelStatisticsImageFilter()
                stats_filter.Execute(ccf_labelmap, ccf_labelmap)
                label_values = list(stats_filter.GetLabels())
                label_values.remove(0)
                n_qanats += len(label_values)
                print('label file: {}, no of qanats: {}'.format(os.path.basename(label_file), len(label_values)))
                for label_value in tqdm(label_values):
                    shape_stats = get_label_shape_stats(ccf_labelmap, label_value)
                    d.append(OrderedDict({
                        'data source': data_source,
                        'corona file name': corona_folder.split('/')[-1],
                        'label file name': os.path.basename(label_file),
                        'label value': label_value,
                        'centroid': shape_stats['centroid'],
                        'bounding box': shape_stats['bounding_box'],
                        'pixel nos': shape_stats['pixel_nos'],
                        'elongation': shape_stats['elongation'],
                        'roundness': shape_stats['roundness'],
                        'ccf file path': outpu_ccf_file_path,
                    }))
        print('for data source {} there are {} qanats.'.format(data_source, n_qanats))
    pd.DataFrame(d).to_csv(os.path.join(sheets_folder, 'third_attempt_qanat_stats.csv'))
