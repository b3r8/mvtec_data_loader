# MVTEC Data Loader

Data loader for the MVTec dataset, a comprehensive real-world dataset for Unsupervised Anomaly Detection

MVTec dataset paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.pdf

The MVTEC dataset is available at: https://www.mvtec.com/company/research/datasets/mvtec-ad

## How to use it

The 'mvtecDataset.py' file and the mvtec data directory must be in the same directory, such that:

/.../your_directory/mvtecDataset.py

/.../your_directory/mvtec/bottle/...

/.../your_directory/mvtec/cable/...

and so on

The images in the dataset are high resolution, so you can resize them with the 'resize' option. For more details, please see the example provided (notebook)
