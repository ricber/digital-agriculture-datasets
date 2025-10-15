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
- **Plant Seedlings Dataset**: The Plant Seedlings Dataset contains 5,539 images images featuring plants belonging to 12 species at several growth stages - Data: https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset/data - Paper: https://arxiv.org/abs/1711.05458

## Semantic segmentation
### Crop and Weed datasets
- **Sugar Beets 2016**: https://www.ipb.uni-bonn.de/data/sugarbeets2016/
- **WeedMap**: A Large-Scale Semantic Segmentation Crop-Weed Dataset Using Aerial Color and Multispectral Imaging - https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap
- **WE3DS**: A dataset of RGB-D images acquired through a ground vehicle, with 2,568 annotated images containing 17 plant species (7 crops + 10 weeds) - https://zenodo.org/records/7457983
### Plant Instance and Plant Part datasets
These datasets provide pixel-wise labels to distinguish between different plant parts, or between plants and soil.
- **VegAnn: Vegetation Annotation of a large multi-crop RGB Dataset acquired under diverse conditions for image segmentation**: VegAnn contains 3775 labeled images (512*512 pixels) with two clases (Background and Vegetation). The dataset includes images of 26+ crop species. - Dataset: https://zenodo.org/records/7636408 - Paper: https://www.nature.com/articles/s41597-023-02098-y
- **The Capsicum annuum dataset**: This dataset consists of per-pixel annotated synthetic (10500) and empirical images (50) of Capsicum annuum, also known as sweet or bell pepper, situated in a commercial greenhouse. Furthermore, the source models to generate the synthetic images are included. Data: https://doi.org/10.4121/uuid:884958f5-b868-46e1-b3d8-a0b5d91b02c0 - Paper: https://doi.org/10.1016/j.compag.2017.12.001
### Disease and Plant Health datasets
- **RoCoLe (Robusta Coffee Leaf Images)**: 1,560 high-resolution images of coffee leaves, annotated for semantic segmentation tasks. Includes 2,329 labeled objects across 7 classes: healthy, unhealthy, rust_level_1, rust_level_2, rust_level_3, rust_level_4, and red_spider_mite. Captured under real-world conditions using a smartphone camera - https://doi.org/10.17632/c5yvn32dzg.2
### Scene understanding datasets (2D & 3D Semantic Segmentation)
These datasets enable holistic scene understanding, including semantic segmentation in both 2D images and 3D point clouds. They are particularly relevant for robotics in natural, off-road, or forested environments.
- **RELLIS-3D: A Multi-modal Dataset for Off-Road Robotics**: Semantic segmentation on 2D RGB images and **3D LiDAR pointclouds** - https://github.com/unmannedlab/RELLIS-3D/tree/main
- **RUGD Dataset**: The RUGD dataset focuses on semantic understanding of unstructured outdoor environments for applications in off-road autonomous navigation. The datset is comprised of video sequences captured from the camera onboard a mobile robot platform - http://rugd.vision/
- **GOOSE dataset**: GOOSE is the German Outdoor and Offroad Dataset and is a 2D & 3D semantic segmentation dataset framework. In contrast to existing datasets like Cityscapes or BDD100K, the focus is on unstructured off-road environments - https://goose-dataset.de/docs/
- **WildScenes**: The WildScenes dataset is a multi-modal collection of traversals within Australian forests. The dataset is divided into five sequences across two forest locations. These sequences are both across different physical locations and across different times - https://csiro-robotics.github.io/WildScenes/
- **BotanicGarden**: A robot navigation dataset in a botanic garden of more than 48000m2. Comprehensive sensors are used, including Gray and RGB stereo cameras, spinning and MEMS 3D LiDARs, and low-cost and industrial-grade IMUs. An all-terrain wheeled robot is employed for data collection, traversing through thick woods, riversides, narrow trails, bridges, and grasslands. This yields 33 short and long sequences, forming 17.1km trajectories in total - https://github.com/robot-pesg/BotanicGarden

## Object detection
These datasets support bounding box annotations for tasks like fruit counting, weed identification, or tracking crop objects across frames.
### Crop and Weed datasets
- **WeedCrop Image Dataset**: It includes 2822 images annotated in YOLO v5 PyTorch format - https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset
- **CornWeed Dataset: A dataset for training maize and weed object detectors for agricultural machines**: https://zenodo.org/records/7961764
- **The Dataset of annotated food crops and weed**: The dataset contains 1,118 images and 7,853 manual annotations of food crops and weeds in their early seedling stages. The dataset was collected in several locations in Latvia and describes eight weed and six food species - Data: http://doi.org/10.17632/nj4vtk4tt6.1 - Paper: http://doi.org/10.1016/j.dib.2020.105833
- **Sesame Crop and Weed Detection** — Dataset with 1,300 RGB images (512×512) of sesame crops and weeds, including 2,072 annotated bounding boxes for object detection in precision agriculture - https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes
### Fruit datasets
- **Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking**: https://www.sciencedirect.com/science/article/pii/S2352340922010514
- **ACFR Orchard Fruit Dataset**: https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/
- **GrapeMOTS**: UAV vineyard dataset with MOTS grape bunch annotations recorded from multiple perspectives for enhanced object detection and tracking - https://doi.org/10.1016/j.dib.2024.110432
- **CitDet**: A Benchmark Dataset for Citrus Fruit Detection - https://robotic-vision-lab.github.io/citdet/
- **deepNIR Fruit Detection**: 4,295 RGB + (synthetic) NIR images annotated with 161,979 bounding boxes over 11 fruit / crop classes (apple, avocado, capsicum, mango, orange, rockmelon, strawberry, blueberry, cherry, kiwi, wheat) - https://doi.org/10.5281/zenodo.6324489
- **Strawberry Dataset for Object Detection**: 813 images with 4,568 labeled objects across 3 classes (ripe, peduncle, unripe), annotated for object detection tasks in strawberry harvesting automation - https://doi.org/10.5281/zenodo.6126677
- **Apple Dataset Benchmark from Orchard Environment**: 2,299 images with 15,439 bounding box annotations of apples, captured under various lighting and seasonal conditions using modern fruiting wall architecture - https://doi.org/10.7273/000001752
- **Tomato Detection Dataset**: 895 images with 4,930 labeled tomatoes, annotated for object detection tasks in greenhouse environments - https://www.kaggle.com/datasets/andrewmvd/tomato-detection
### Disease and Plant Health datasets
- **Rice Disease Dataset**: 470 images with 1,956 bounding box annotations across 3 classes (Bacterial Blight, Brown Spot, Rice Blast) - https://www.kaggle.com/dsv/2481060
- **Multispectral Potato Plants Images**: 360 RGB image patches (750×750 px) with bounding box annotations for object detection tasks in potato crop health assessment - https://datasetninja.com/multispectral-potato-plants-images

## Instance segmentation (detection + segmentation)
These datasets provide both bounding boxes and segmentation masks for precise object identification, often with species-level annotations.
### Crop and Weed datasets
- **The Sorghum Weed Dataset**: The dataset contains 5555 manually pixel-wise annotated instances from 252 images which contain sorghum, grass weed, and broadleaf weed samplings that can be used for object detection, instance segmentation, and semantic segmentation - https://doi.org/10.1016/j.dib.2023.109935
- **The ACRE Crop-Weed Dataset**: Crop-weed dataset containing 1000 RGB images featuring two crop species (maize and bean) and four weed species - https://zenodo.org/records/8102217
- **ROSE Challenge dataset**: Crop-weed dataset containing RGB images collected over three different years by four different robots, featuring two crop species (maize and bean) and four weed species - https://www.challenge-rose.fr/en/dataset-download/
  - **WeedElec team images**: This is an extra set of 83 field images containing 2489 instances of crop and weed specimens collected and annotated by the WeedElec team that participated in the ROSE challenge - Data: https://zenodo.org/records/3906501 - Paper: https://doi.org/10.1002/aps3.11373
- **The CropAndWeed Dataset**: 8k high-quality images and about 112k annotated plant instances. In addition to bounding boxes, segmentation masks and stem positions, annotations include a fine-grained classification into 16 crop and 58 weed species, as well as extensive meta-annotations of relevant environmental and recording parameters - https://github.com/cropandweed/cropandweed-dataset/tree/main
### Fruit datasets
- **Embrapa Wine Grape Instance Segmentation Dataset – Embrapa WGISD**: https://github.com/thsant/wgisd
- **MinneApple: A Benchmark Dataset for Apple Detection and Segmentation**: https://github.com/nicolaihaeni/MinneApple
- **Sweet Pepper Dataset**: 620 RGB-D images (1280×720 px) with 6,422 labeled objects across 8 classes (green fruit, red fruit, green peduncle, yellow fruit, red peduncle, yellow peduncle, orange fruit, orange peduncle), annotated for instance segmentation, semantic segmentation, and object detection tasks in greenhouse environments - https://www.kaggle.com/datasets/lemontyc/sweet-pepper/data
### Plant Instance and Plant Part datasets
- **Paddy Rice Imagery Dataset for Panicle Segmentation**: 400 high-resolution UAV images (4096×2160 px) with 51,730 pixel-level annotations of rice panicles, captured during the heading, flowering, and ripening stages. Includes manual and semi-supervised annotations - https://doi.org/10.5281/zenodo.4444741

## Tracking
Tracking datasets provide annotated object trajectories across frames, useful for temporal consistency in detection and behavior prediction.
- **Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking**: https://www.sciencedirect.com/science/article/pii/S2352340922010514
- **GrapeMOTS: UAV vineyard dataset with MOTS grape bunch annotations recorded from multiple perspectives for enhanced object detection and tracking**: https://doi.org/10.1016/j.dib.2024.110432

## Hyperspectral and Multispectral imaging
- **CitrusFarm Dataset**: CitrusFarm is a multimodal agricultural robotics dataset that provides both multispectral images and navigational sensor data for localization, mapping and crop monitoring tasks - https://ucr-robotics.github.io/Citrus-Farm-Dataset/
- **Multispectral Potato Plants Images**: 360 RGB image patches (750×750 px) with bounding box annotations for object detection tasks in potato crop health assessment - https://datasetninja.com/multispectral-potato-plants-images

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
- **Dataset Ninja**: https://datasetninja.com/category/agriculture
- **Quantitative Plant**: Website that collects datasets for image classification, semantic segmentation and phenotyping - https://www.quantitative-plant.org/dataset
- **A survey of public datasets for computer vision tasks in precision agriculture**: Collection of datasets for detection and segmentation of weeds and fruits and phenotyping tasks (e.g., damage and disease detection, biomas prediction, yield estimation) - https://doi.org/10.1016/j.compag.2020.105760
- **Weed database development: An updated survey of public weed datasets and cross-season weed detection adaptation**: A survey of 36 publicly available image datasets for weed recognition (classification, detection, and segmentation) + a new two-season dataset of eight weed classes curated for cross-season modeling - https://doi.org/10.1016/j.ecoinf.2024.102546

## Tools to create synthetic datasets
- **CropCraft**: CropCraft is a python script that generates 3D models of crop fields, specialized in real-time simulation of robotics applications - https://github.com/Romea/cropcraft
- **TomatoSynth**: TomatoSynth provides realistic synthetic tomato plants training data for deep learning applications, reducing the need for manual annotation and allowing customization for specific greenhouse environments, thus advancing automation in agriculture - https://github.com/SCT-lab/TomatoSynth

## Contributing
Contributions are very welcome! 🚜🌱  
If you know of additional datasets, tools, or surveys that fit into this list, please feel free to open an issue or submit a pull request.
Just make sure to include a short description with the dataset size (number of samples, number of labels, etc.), the types of data, and a valid link so the resource can be easily integrated.
