# Digital Agriculture Datasets
This curated list gathers publicly available datasets supporting the development of artificial intelligence and robotics solutions for agriculture. The datasets are organized by task—ranging from image classification to robotic navigation—and span different sensing modalities, including RGB, multispectral, hyperspectral, and 3D LiDAR. These resources are useful for training and evaluating machine learning models, especially in precision agriculture, crop monitoring, weed detection, and off-road autonomous navigation.

## Image classification
Datasets in this section are designed for tasks like plant species identification or detection of crop diseases and nutrient deficiencies using single-label classification of images.
- **PlantCLEF2022**: Image-based plant identification at global scale - https://www.imageclef.org/plantclef2022
- **Deep Learning for Non-Invasive Diagnosis of Nutrient Deficiencies in Sugar Beet Using RGB Images**: https://zenodo.org/records/4106221#.YqdMcexBzon
- **Weed25: A deep learning dataset for weed identification**: https://doi.org/10.3389/fpls.2022.1053329
- **A phenotyping weeds image dataset for open scientific research**: https://zenodo.org/records/7598372
- **SorghumWeedDataset_Classification and SorghumWeedDataset_Segmentation datasets for classification, detection, and segmentation in deep learning**: https://doi.org/10.1016/j.dib.2023.109935
- **DeepWeeds**: A multiclass weed species image dataset consisting of 17,509 images capturing eight different weed species native to Australia in situ with neighbouring flora - https://github.com/AlexOlsen/DeepWeeds
- **PlantVillage Dataset**: 50,000 expertly curated images on healthy and infected leaves of crops plants through the existing online platform PlantVillage - https://github.com/spMohanty/PlantVillage-Dataset

## Semantic segmentation
### Plant Instance & Part Segmentation
These datasets provide pixel-wise labels to distinguish between different plant parts, between crops and weeds, or between crop and soil.
- **Sugar Beets 2016**: https://www.ipb.uni-bonn.de/data/sugarbeets2016/
- **A Crop/Weed Field Image Dataset for the Evaluation of Computer Vision Based Precision Agriculture Tasks**: 60 RGB annotated images - https://github.com/cwfid/dataset/tree/master
- **WeedMap**: A Large-Scale Semantic Segmentation Crop-Weed Dataset Using Aerial Color and Multispectral Imaging - https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap
- **WE3DS**: A dataset of RGB-D images acquired through a ground vehicle, with 2,568 densely annotated semantic label maps containing 17 plant species (7 crops + 10 weeds) - https://zenodo.org/records/7457983
- **VegAnn: Vegetation Annotation of a large multi-crop RGB Dataset acquired under diverse conditions for image segmentation**: VegAnn contains 3775 labeled images (512*512 pixels) with two clases (Background and Vegetation). The dataset includes images of 26+ crop species. - Dataset: https://zenodo.org/records/7636408 - Paper: https://www.nature.com/articles/s41597-023-02098-y
- **Plant Seedlings Dataset**: The Plant Seedlings Dataset contains images of approximately 960 unique plants belonging to 12 species at several growth stages. It comprises annotated RGB images for classification and segmentation tasks - https://vision.eng.au.dk/plant-seedlings-dataset/
#### Synthetic datasets for Plant Instance & Part Segmentation
- **Data synthesis methods for semantic segmentation in agriculture: A Capsicum annuum dataset**: https://doi.org/10.1016/j.compag.2017.12.001
### Scene understanding (2D & 3D Semantic Segmentation)
These datasets enable holistic scene understanding, including semantic segmentation in both 2D images and 3D point clouds. They are particularly relevant for robotics in natural, off-road, or forested environments.
- **RELLIS-3D: A Multi-modal Dataset for Off-Road Robotics**: Semantic segmentation on 2D RGB images and **3D LiDAR pointclouds** - https://github.com/unmannedlab/RELLIS-3D/tree/main
- **RUGD Dataset**: The RUGD dataset focuses on semantic understanding of unstructured outdoor environments for applications in off-road autonomous navigation. The datset is comprised of video sequences captured from the camera onboard a mobile robot platform - http://rugd.vision/
- **GOOSE dataset**: GOOSE is the German Outdoor and Offroad Dataset and is a 2D & 3D semantic segmentation dataset framework. In contrast to existing datasets like Cityscapes or BDD100K, the focus is on unstructured off-road environments - https://goose-dataset.de/docs/
- **WildScenes**: The WildScenes dataset is a multi-modal collection of traversals within Australian forests. The dataset is divided into five sequences across two forest locations. These sequences are both across different physical locations and across different times - https://csiro-robotics.github.io/WildScenes/
- **BotanicGarden**: A robot navigation dataset in a botanic garden of more than 48000m2. Comprehensive sensors are used, including Gray and RGB stereo cameras, spinning and MEMS 3D LiDARs, and low-cost and industrial-grade IMUs. An all-terrain wheeled robot is employed for data collection, traversing through thick woods, riversides, narrow trails, bridges, and grasslands. This yields 33 short and long sequences, forming 17.1km trajectories in total - https://github.com/robot-pesg/BotanicGarden

## Object detection
These datasets support bounding box annotations for tasks like fruit counting, weed identification, or tracking crop objects across frames.
- **Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking**: https://www.sciencedirect.com/science/article/pii/S2352340922010514
- **ACFR Orchard Fruit Dataset**: https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/
- **GrapeMOTS: UAV vineyard dataset with MOTS grape bunch annotations recorded from multiple perspectives for enhanced object detection and tracking**: https://doi.org/10.1016/j.dib.2024.110432
- **CornWeed Dataset: A dataset for training maize and weed object detectors for agricultural machines**: https://zenodo.org/records/7961764
- **CitDet: A Benchmark Dataset for Citrus Fruit Detection**: https://robotic-vision-lab.github.io/citdet/
- **WeedCrop Image Dataset**: It includes 2822 images annotated in YOLO v5 PyTorch format - https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset

## Instance segmentation (detection + segmentation)
These datasets provide both bounding boxes and segmentation masks for precise object identification, often with species-level annotations.
### Crop and Weed datasets
- **SorghumWeedDataset_Classification and SorghumWeedDataset_Segmentation datasets for classification, detection, and segmentation in deep learning**: https://doi.org/10.1016/j.dib.2023.109935
- **The ACRE Crop-Weed Dataset**: Crop-weed dataset containing 1000 RGB images featuring two crop species (maize and bean) and four weed species - https://zenodo.org/records/8102217
- **ROSE Challenge dataset**: Crop-weed dataset containing RGB images collected over three different years by four different robots, featuring two crop species (maize and bean) and four weed species - https://www.challenge-rose.fr/en/dataset-download/
  - **WeedElec team images**: This is an extra set of 83 field images containing 2489 instances of crop and weed specimens collected and annotated by the WeedElec team that participated in the ROSE challenge - Data: https://zenodo.org/records/3906501 - Paper: https://doi.org/10.1002/aps3.11373
- **The CropAndWeed Dataset: A Multi-Modal Learning Approach for Efficient Crop and Weed Manipulation**: 8k high-quality images and about 112k annotated plant instances. In addition to bounding boxes, segmentation masks and stem positions, annotations include a fine-grained classification into 16 crop and 58 weed species, as well as extensive meta-annotations of relevant environmental and recording parameters - https://github.com/cropandweed/cropandweed-dataset/tree/main
### Fruit datasets
- **Embrapa Wine Grape Instance Segmentation Dataset – Embrapa WGISD**: https://github.com/thsant/wgisd
- **MinneApple: A Benchmark Dataset for Apple Detection and Segmentation**: https://github.com/nicolaihaeni/MinneApple

## Tracking
Tracking datasets provide annotated object trajectories across frames, useful for temporal consistency in detection and behavior prediction.
- **Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking**: https://www.sciencedirect.com/science/article/pii/S2352340922010514
- **GrapeMOTS: UAV vineyard dataset with MOTS grape bunch annotations recorded from multiple perspectives for enhanced object detection and tracking**: https://doi.org/10.1016/j.dib.2024.110432

## Hyperspectral imaging
- **CitrusFarm Dataset**: CitrusFarm is a multimodal agricultural robotics dataset that provides both multispectral images and navigational sensor data for localization, mapping and crop monitoring tasks - https://ucr-robotics.github.io/Citrus-Farm-Dataset/

## Robotics
These datasets support autonomous navigation, localization, and mapping in agriculture and forestry. They usually contain unlabeled multimodal data from a variety of sensors.
- **Sugar Beets 2016**: https://www.ipb.uni-bonn.de/data/sugarbeets2016/
- **CitrusFarm Dataset**: CitrusFarm is a multimodal agricultural robotics dataset that provides both multispectral images and navigational sensor data for localization, mapping and crop monitoring tasks - https://ucr-robotics.github.io/Citrus-Farm-Dataset/
- **A high-resolution, multimodal data set for agricultural robotics: A Ladybird's-eye view of Brassica**: https://doi.org/10.1002/rob.21877
- **RELLIS-3D: A Multi-modal Dataset for Off-Road Robotics**: Semantic segmentation on 2D RGB images and **3D LiDAR pointclouds** - https://github.com/unmannedlab/RELLIS-3D/tree/main
- **RUGD Dataset**: The RUGD dataset focuses on semantic understanding of unstructured outdoor environments for applications in off-road autonomous navigation. The datset is comprised of video sequences captured from the camera onboard a mobile robot platform. - http://rugd.vision/
- **GOOSE dataset**: GOOSE is the German Outdoor and Offroad Dataset and is a 2D & 3D semantic segmentation dataset framework. In contrast to existing datasets like Cityscapes or BDD100K, the focus is on unstructured off-road environments - https://goose-dataset.de/docs/
- **WildScenes**: The WildScenes dataset is a multi-modal collection of traversals within Australian forests. The dataset is divided into five sequences across two forest locations. These sequences are both across different physical locations and across different times - https://csiro-robotics.github.io/WildScenes/
- **BotanicGarden**: A robot navigation dataset in a botanic garden of more than 48000m2. Comprehensive sensors are used, including Gray and RGB stereo cameras, spinning and MEMS 3D LiDARs, and low-cost and industrial-grade IMUs. An all-terrain wheeled robot is employed for data collection, traversing through thick woods, riversides, narrow trails, bridges, and grasslands. This yields 33 short and long sequences, forming 17.1km trajectories in total - https://github.com/robot-pesg/BotanicGarden
- **Rosario**: A multi-modal dataset collected in a soybean crop field, comprising over two hours of recorded data from sensors such as stereo infrared camera, color camera, accelerometer, gyroscope, magnetometer, GNSS (Single Point Positioning, Real-Time Kinematic and Post-Processed Kinematic), and wheel odometry. This dataset captures key challenges inherent to robotics in agricultural environments - https://cifasis.github.io/rosariov2/
- **BLT (Bacchus Long Term)**: A multi-session agricultural field dataset collected over months in vineyards (Greece and the UK), featuring seasonal variation, repeated traversal paths and onboard multimodal sensing (RGB-D, LiDAR, navigation). It is designed to support long-term mapping, localisation, crop phenotyping and generalisation studies in robotic agriculture - https://lcas.lincoln.ac.uk/wp/research/data-sets-software/blt/

## Collectors of datasets
- **Quantitative Plant**: Website that collects datasets for image classification, semantic segmentation and phenotyping - https://www.quantitative-plant.org/dataset
- **A survey of public datasets for computer vision tasks in precision agriculture**: Collection of datasets for detection and segmentation of weeds and fruits and phenotyping tasks (e.g., damage and disease detection, biomas prediction, yield estimation) - https://doi.org/10.1016/j.compag.2020.105760
- **Weed database development: An updated survey of public weed datasets and cross-season weed detection adaptation**: A survey of 36 publicly available image datasets for weed recognition (classification, detection, and segmentation) + a new two-season dataset of eight weed classes curated for cross-season modeling - https://doi.org/10.1016/j.ecoinf.2024.102546

## Tools to create synthetic datasets
- **CropCraft**: CropCraft is a python script that generates 3D models of crop fields, specialized in real-time simulation of robotics applications - https://github.com/Romea/cropcraft
- **TomatoSynth**: TomatoSynth provides realistic synthetic tomato plants training data for deep learning applications, reducing the need for manual annotation and allowing customization for specific greenhouse environments, thus advancing automation in agriculture - https://github.com/SCT-lab/TomatoSynth

## Contributing
Contributions are very welcome! 🚜🌱  
If you know of additional datasets, tools, or surveys that fit into this list, please feel free to open an issue or submit a pull request.  
Just make sure to provide a short description and a valid link so that the resource can be easily integrated.
