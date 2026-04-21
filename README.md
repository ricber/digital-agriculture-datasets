# Digital Agriculture Datasets
This curated list gathers publicly available datasets supporting the development of artificial intelligence and robotics solutions for agriculture. The datasets are organized by task—ranging from image classification to robotic navigation—and span different sensing modalities, including RGB, multispectral, hyperspectral, and 3D LiDAR. These resources are useful for training and evaluating machine learning models, especially in precision agriculture, crop monitoring, weed detection, and off-road autonomous navigation.

## Contributing
Contributions are very welcome! 🚜🌱  
If you know of additional datasets, tools, or surveys that fit into this list, please feel free to open an issue or submit a pull request.
Just make sure to include a short description with the dataset size (number of samples, number of labels, etc.), the types of data, and a valid link so the resource can be easily integrated.

---

## 📖 Index

1. [Image Classification](#image-classification)  
2. [Semantic Segmentation](#semantic-segmentation)  
   - [Crop and Weed](#crop-and-weed-datasets)  
   - [Plant Instance and Plant Part](#plant-instance-and-plant-part-datasets)  
   - [Disease and Plant Health](#disease-and-plant-health-datasets)  
   - [Scene Understanding (2D & 3D)](#scene-understanding-datasets-2d--3d-semantic-segmentation)  
3. [Object Detection](#object-detection)  
   - [Crop and Weed](#crop-and-weed-datasets-1)  
   - [Plant Instance and Plant Part](#plant-instance-and-plant-part-datasets-1)  
   - [Disease and Plant Health](#disease-and-plant-health-datasets-1)  
4. [Instance Segmentation](#instance-segmentation-detection--segmentation)  
   - [Crop and Weed](#crop-and-weed-datasets-2)  
   - [Plant Instance and Plant Part](#plant-instance-and-plant-part-datasets-2)
5. [Large-Scale and Unlabeled Image Datasets](#large-scale-and-unlabeled-image-datasets)
6. [3D Plant and Point Cloud Datasets](#3d-plant-and-point-cloud-datasets)
7. [Tracking](#tracking)  
8. [Hyperspectral and Multispectral Imaging](#hyperspectral-and-multispectral-imaging)  
9. [Robotics](#robotics)
10. [Collectors of Datasets](#collectors-of-datasets)  
11. [Tools to Create Synthetic Datasets](#tools-to-create-synthetic-datasets)  

---

## Image Classification

Datasets in this section are designed for tasks like plant species identification or detection of crop diseases and nutrient deficiencies using single-label classification.

- **PlantCLEF2022** — Image-based plant identification at global scale. [Data](https://www.imageclef.org/plantclef2022)  
- **Deep Learning for Non-Invasive Diagnosis of Nutrient Deficiencies in Sugar Beet Using RGB Images** — [Data](https://zenodo.org/records/4106221#.YqdMcexBzon)  
- **Weed25** — Deep learning dataset for weed identification. [Paper](https://doi.org/10.3389/fpls.2022.1053329)  
- **Phenotyping Weeds Image Dataset** — Open scientific research dataset for weed phenotyping. [Data](https://zenodo.org/records/7598372)  
- **The Sorghum Weed Classification Dataset** — 4,312 samples for crop-weed classification problems. [Paper](https://doi.org/10.1016/j.dib.2023.109935)  
- **DeepWeeds** — 17,509 images of 8 Australian weed species in situ. [Data](https://github.com/AlexOlsen/DeepWeeds)  
- **PlantVillage Dataset** — 50,000 images of healthy and infected crop leaves. [Data](https://github.com/spMohanty/PlantVillage-Dataset)  
- **Plant Seedlings Dataset** — 5,539 images of 12 species at different growth stages. [Data](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset/data), [Paper](https://arxiv.org/abs/1711.05458)
- **Pl@ntNet‑300K** — A large-scale plant image collection (~ 306,146 images) covering 1,081 plant species, curated from citizen science observations — [Data](https://doi.org/10.5281/zenodo.4726653), [Paper](https://openreview.net/forum?id=eLYinD0TtIt)
- **iNatAg** — A large‑scale agricultural image dataset comprising over 4.7 million images of 2,959 crop and weed species, with hierarchical labels (binary crop/weed up to species level), drawn globally from iNaturalist. Enables multi‑class and taxonomic classification in agricultural domains — [Data](https://github.com/Project-AgML/AgML), [Paper](https://arxiv.org/abs/2503.20068)

---

## Semantic segmentation
### Crop and Weed datasets
- **Sugar Beets 2016**: https://www.ipb.uni-bonn.de/data/sugarbeets2016/
- **WeedMap**: A Large-Scale Semantic Segmentation Crop-Weed Dataset Using Aerial Color and Multispectral Imaging - https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap
- **WE3DS**: A dataset of RGB-D images acquired through a ground vehicle, with 2,568 annotated images containing 17 plant species (7 crops + 10 weeds) - https://zenodo.org/records/7457983
### Plant Instance and Plant Part datasets
These datasets provide pixel-wise labels to distinguish between different plant parts, or between plants and soil.
- **VegAnn**: VegAnn is a large multi-crop RGB Dataset acquired under diverse conditions for image segmentation. It contains 3775 labeled images (512*512 pixels) with two clases (Background and Vegetation). The dataset includes images of 26+ crop species. - Dataset: https://zenodo.org/records/7636408 - Paper: https://www.nature.com/articles/s41597-023-02098-y
- **Plant Growth Segmentation**: 2,008 high-resolution images with pixel-level semantic segmentation annotations of plant regions, captured over multiple growth stages. - https://www.kaggle.com/datasets/shengyou222/plantgrowthsegmentationdatase
- **RiceSEG (Global Rice Multi-Class Segmentation Dataset)** — An RGB image dataset comprising images collected from five major rice-growing countries (China, Japan, India, Philippines, Tanzania), covering more than 6,000 rice genotypes over all growth stages. It contains 3,078 images annotated into six classes: background, green vegetation, senescent vegetation, panicle, weeds, and duckweed. [Data](https://www.global-rice.com), [Paper](https://www.sciencedirect.com/science/article/pii/S2643651525001050)
- **The Capsicum annuum dataset**: This dataset consists of per-pixel annotated synthetic (10500) and empirical images (50) of Capsicum annuum, also known as sweet or bell pepper, situated in a commercial greenhouse. Furthermore, the source models to generate the synthetic images are included. Data: https://doi.org/10.4121/uuid:884958f5-b868-46e1-b3d8-a0b5d91b02c0 - Paper: https://doi.org/10.1016/j.compag.2017.12.001
### Disease and Plant Health datasets
- **RoCoLe (Robusta Coffee Leaf Images)**: 1,560 high-resolution images of coffee leaves, annotated for semantic segmentation tasks. Includes 2,329 labeled objects across 7 classes: healthy, unhealthy, rust_level_1, rust_level_2, rust_level_3, rust_level_4, and red_spider_mite. Captured under real-world conditions using a smartphone camera - https://doi.org/10.17632/c5yvn32dzg.2
### Scene understanding datasets (2D & 3D Semantic Segmentation)
These datasets enable holistic scene understanding, including semantic segmentation in both 2D images and 3D point clouds. They are particularly relevant for robotics in natural, off-road, or forested environments.
- **RELLIS-3D: A Multi-modal Dataset for Off-Road Robotics**: Semantic segmentation on 2D RGB images and **3D LiDAR pointclouds** - https://github.com/unmannedlab/RELLIS-3D/tree/main
- **RUGD Dataset**: The RUGD dataset focuses on semantic understanding of unstructured outdoor environments for applications in off-road autonomous navigation. The datset is comprised of video sequences captured from the camera onboard a mobile robot platform - http://rugd.vision/
- **GOOSE dataset**: GOOSE is the German Outdoor and Offroad Dataset and is a 2D & 3D semantic segmentation dataset framework. In contrast to existing datasets like Cityscapes or BDD100K, the focus is on unstructured off-road environments - https://goose-dataset.de/docs/
- **WildScenes**: The WildScenes dataset is a multi-modal collection of traversals within Australian forests. The dataset is divided into five sequences across two forest locations. These sequences are both across different physical locations and across different times - https://csiro-robotics.github.io/WildScenes/
- **BotanicGarden**: A robot navigation dataset in a botanic garden of more than 48000m2. Comprehensive sensors are used, including Gray and RGB stereo cameras, spinning and MEMS 3D LiDARs, and low-cost and industrial-grade IMUs. An all-terrain wheeled robot is employed for data collection, traversing through thick woods, riversides, narrow trails, bridges, and grasslands. This yields 33 short and long sequences, forming 17.1km trajectories in total - [Data](https://github.com/robot-pesg/BotanicGarden)
- **Crop Row Detection Lincoln Dataset (CRDLD)** — A crop-row detection / semantic segmentation dataset for agricultural robot navigation in maize and sugar beet fields. CRDLD v2.1 contains 2,000 field images (1,250 train / 500 test / 250 val), each paired with ground-truth row labels and MATLAB `.mat` files with labeled coordinates. The dataset spans 50 field-condition classes, including shadows, varying crop growth, weed density, discontinuities, slopes/curves, and tyre tracks. [Data](https://github.com/JunfengGaolab/CropRowDetection), [Paper](https://doi.org/10.1002/rob.22238)

## Object detection
These datasets support bounding box annotations for tasks like fruit counting, weed identification, or tracking crop objects across frames.
### Crop and Weed datasets
- **WeedCrop Image Dataset**: It includes 2822 images annotated in YOLO v5 PyTorch format - https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset
- **CornWeed Dataset: A dataset for training maize and weed object detectors for agricultural machines**: https://zenodo.org/records/7961764
- **The Dataset of annotated food crops and weed**: The dataset contains 1,118 images and 7,853 manual annotations of food crops and weeds in their early seedling stages. The dataset was collected in several locations in Latvia and describes eight weed and six food species - Data: http://doi.org/10.17632/nj4vtk4tt6.1 - Paper: http://doi.org/10.1016/j.dib.2020.105833
- **Sesame Crop and Weed Detection** — Dataset with 1,300 RGB images (512×512) of sesame crops and weeds, including 2,072 annotated bounding boxes for object detection in precision agriculture - https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes
- **WeedMaize Dataset**: 7,784 images with 121,635 bounding-box annotations across 18 classes (e.g., maize, Cyperus rotundus, Solanum nigrum, Echinochloa crus-galli). - https://doi.org/10.5281/zenodo.5106795
- **CottonWeedDet3**: 848 high-resolution RGB images with 1,532 bounding-box annotations across 3 weed classes (morningglory, carpetweed, palmer amaranth) in cotton cropping systems. [Data](https://doi.org/10.34740/KAGGLE/DSV/4090494)
- **DeepSeedling**: 5,741 images with 33,255 bounding-box annotations across 2 classes (plant, weed), captured in cotton fields under varying conditions. [Data](https://figshare.com/s/616956f8633c17ceae9b)
### Plant Instance and Plant Part datasets
- **Vineyard UAV Datasets - Tomiño, Pontevedra, Galicia**
  - **UAV RGB Vineyard Dataset with Bunch Labels** — Videos with annotated grape bunches for object detection and tracking collected in 2021. [Data](https://doi.org/10.5281/zenodo.7330951), [Paper](https://www.sciencedirect.com/science/article/pii/S2352340922010514)
  - **GrapeMOTS** — UAV dataset with MOTS annotations of grape bunches, recorded from multiple perspectives for enhanced object detection and tracking. Collected in 2023. [Data](https://doi.org/10.5281/zenodo.10625595), [Paper](https://doi.org/10.1016/j.dib.2024.110432)
- **ACFR Orchard Fruit Dataset**: https://data.acfr.usyd.edu.au/ag/treecrops/2016-multifruit/
- **CitDet**: A Benchmark Dataset for Citrus Fruit Detection - https://robotic-vision-lab.github.io/citdet/
- **deepNIR Fruit Detection**: 4,295 RGB + (synthetic) NIR images annotated with 161,979 bounding boxes over 11 fruit / crop classes (apple, avocado, capsicum, mango, orange, rockmelon, strawberry, blueberry, cherry, kiwi, wheat) - https://doi.org/10.5281/zenodo.6324489
- **Strawberry Dataset for Object Detection**: 813 images with 4,568 labeled objects across 3 classes (ripe, peduncle, unripe), annotated for object detection tasks in strawberry harvesting automation - https://doi.org/10.5281/zenodo.6126677
- **Apple Dataset Benchmark from Orchard Environment**: 2,299 images with 15,439 bounding box annotations of apples, captured under various lighting and seasonal conditions using modern fruiting wall architecture - https://doi.org/10.7273/000001752
- **Tomato Detection Dataset**: 895 images with 4,930 labeled tomatoes, annotated for object detection tasks in greenhouse environments - https://www.kaggle.com/datasets/andrewmvd/tomato-detection
- **Global Wheat Head Detection (GWHD) Dataset**: 4,948 high-resolution RGB images with 188,500 labeled wheat heads, captured across 12 countries and various developmental stages. - https://doi.org/10.5281/zenodo.5092309
### Disease and Plant Health datasets
- **Rice Disease Dataset**: 470 images with 1,956 bounding box annotations across 3 classes (Bacterial Blight, Brown Spot, Rice Blast) - https://www.kaggle.com/dsv/2481060
- **Multispectral Potato Plants Images**: 360 RGB image patches (750×750 px) with bounding box annotations for object detection tasks in potato crop health assessment - https://datasetninja.com/multispectral-potato-plants-images
- **PlantDoc**: 2,482 images with 8,595 bounding-box annotations across 29 classes, including diseases like Corn Leaf Blight, Tomato Septoria Leaf Spot, and Potato Early Blight. [Data](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset), [Paper](https://doi.org/10.1145/3371158.3371196)

## Instance segmentation (detection + segmentation)
These datasets provide both bounding boxes and segmentation masks for precise object identification, often with species-level annotations.
### Crop and Weed datasets
- **The Sorghum Weed Segmentation Dataset**: The dataset contains 5555 manually pixel-wise annotated instances from 252 images which contain sorghum, grass weed, and broadleaf weed samplings that can be used for object detection, instance segmentation, and semantic segmentation. [Paper](https://doi.org/10.1016/j.dib.2023.109935)
- **Carrot-Weed Dataset**: 39 RGB images with 745 pixel-level annotations of young carrot seedlings and weeds, captured under varying light conditions in Negotino, North Macedonia. Suitable for instance segmentation, semantic segmentation, and object detection tasks. [Data](https://github.com/lameski/rgbweeddetection), [Paper](https://doi.org/10.1007/978-3-319-67597-8_11)
- **Maize and Bean Crop and Weed Datasets - Montoldre, Saint-Pourçain-sur-Sioule, France**
  - **ROSE Challenge dataset**: Crop-weed dataset containing 3000 RGB images collected over three different years (2019, 2020, 2021) by four different robots (BIPBIP, WeedElec, ROSEAU, PEAD), featuring two crop species (maize and bean) and four weed species. [Data](https://www.challenge-rose.fr/en/dataset-download/)
  - **WeedElec team images**: This is an extra set of 83 field images containing 2489 instances of crop and weed specimens collected and annotated by the WeedElec team that participated in the ROSE challenge. [Data](https://zenodo.org/records/3906501), [Paper](https://doi.org/10.1002/aps3.11373)
  - **The ACRE Crop-Weed Dataset**: Crop-weed dataset containing 1000 RGB images featuring two crop species (maize and bean) and four weed species. [Data](https://zenodo.org/records/8102217)  
- **The CropAndWeed Dataset**: 8k high-quality images and about 112k annotated plant instances. In addition to bounding boxes, segmentation masks and stem positions, annotations include a fine-grained classification into 16 crop and 58 weed species, as well as extensive meta-annotations of relevant environmental and recording parameters. [Data](https://github.com/cropandweed/cropandweed-dataset/tree/main)
- **Weed Growth Stage Dataset**: A large weed imagery collection capturing the complete growth progression of 16 weed species over an 11-week developmental cycle, with 203,567 high-resolution annotated images spanning weekly stages and species-specific temporal labels. Designed for growth stage classification, species identification, and temporal modelling. [Data](https://doi.org/10.5281/zenodo.15808623), [Code](https://github.com/taminulislam/weedswin), [Paper](https://doi.org/10.1038/s41598-025-05092-z)
### Plant Instance and Plant Part datasets
- **Embrapa Wine Grape Instance Segmentation Dataset – Embrapa WGISD**: https://github.com/thsant/wgisd
- **MinneApple: A Benchmark Dataset for Apple Detection and Segmentation**: https://github.com/nicolaihaeni/MinneApple
- **Sweet Pepper Dataset**: 620 RGB-D images (1280×720 px) with 6,422 labeled objects across 8 classes (green fruit, red fruit, green peduncle, yellow fruit, red peduncle, yellow peduncle, orange fruit, orange peduncle), annotated for instance segmentation, semantic segmentation, and object detection tasks in greenhouse environments - https://www.kaggle.com/datasets/lemontyc/sweet-pepper/data
- **StrawDI_Db1 (Strawberry Digital Images Dataset)**: 3,100 high-resolution images (1008×756 px) with 17,938 pixel-level annotations of strawberries, captured under real-world conditions in 20 plantations across 150 hectares in Huelva, Spain. - Data: https://strawdi.github.io/ - Paper: https://doi.org/10.1016/j.compag.2020.105736
- **Paddy Rice Imagery Dataset for Panicle Segmentation**: 400 high-resolution UAV images (4096×2160 px) with 51,730 pixel-level annotations of rice panicles, captured during the heading, flowering, and ripening stages. Includes manual and semi-supervised annotations - https://doi.org/10.5281/zenodo.4444741
- **Apple MOTS**: 2,198 high-resolution images with 105,559 pixel-level instance segmentation annotations of apples in MOTS format, captured using UAVs and wearable sensors in an orchard. - https://doi.org/10.5281/zenodo.5939726
- **Synthetic Plants Dataset**: 10,000 synthetic RGB-D images with 326,754 pixel-level instance segmentation annotations across 4 classes: leaf, petiole, stem, and fruit. Suitable for instance segmentation, semantic segmentation, and object detection tasks in plant phenotyping. - https://www.kaggle.com/datasets/harlequeen/synthetic-rgbd-images-of-plants

## Large-Scale and Unlabeled Image Datasets
Collections of agricultural images without manual annotations, useful for pretraining, self-supervised learning, domain adaptation, and large-scale phenotyping studies. These datasets provide broad coverage of crops, growth stages, and environmental conditions.
- **ImAg4Wheat** — A massive annotated/unannotated wheat imagery dataset, aggregating over 2.5 million images across ~2,000 genotypes and ~500 environmental conditions from 10 countries, covering the full growth cycle (2010–2024) — [Data](https://huggingface.co/datasets/PheniX-Lab/ImAg4Wheat), [Model](https://huggingface.co/PheniX-Lab/FoMo4Wheat), [Paper](https://arxiv.org/abs/2509.06907)

## 3D Plant and Point Cloud Datasets
Datasets providing 3D scans, LiDAR, or point clouds of plants for tasks such as segmentation, reconstruction, and phenotyping.
- **Pheno4D: Spatio‑temporal 3D Plant Point Cloud Dataset** - A high‑resolution, multi‑temporal 3D point cloud dataset of maize and tomato plants grown in controlled conditions and scanned daily with sub‑millimeter accuracy, yielding around 260 million labeled 3D points across growth stages. Each point cloud includes manual labels for semantic and instance tasks like leaf and stem segmentation - [Data](https://www.ipb.uni-bonn.de/data/pheno4d/index.html), [Paper](https://doi.org/10.1371/journal.pone.0256340)
- **BonnBeetClouds3D: High‑Resolution 3D Field Point Clouds for Sugar Beet Phenotyping** - A real‑world 3D point cloud dataset of sugar beet breeding trials captured under field conditions using aerial imagery and photogrammetric reconstruction, covering 48 plant varieties with over 186 annotated plants and 2,661 labeled leaves. Each point cloud contains per‑point semantic and instance labels for plants and leaves plus 10,000+ salient leaf keypoints (tips, bases, corners and plant centers) and reference phenotypic measurements (e.g., leaf length/width) - [Data](https://bonnbeetclouds3d.ipb.uni-bonn.de/), [Paper](https://doi.org/10.1109/IROS58592.2024.10802820)

## Tracking
Tracking datasets provide annotated object trajectories across frames, useful for temporal consistency in detection and behavior prediction.
- **Dataset on UAV RGB videos acquired over a vineyard including bunch labels for object detection and tracking**: https://www.sciencedirect.com/science/article/pii/S2352340922010514
- **GrapeMOTS**: UAV vineyard dataset with MOTS grape bunch annotations recorded from multiple perspectives for enhanced object detection and tracking - Data: https://doi.org/10.5281/zenodo.10625595 - Paper: https://doi.org/10.1016/j.dib.2024.110432
- **Apple MOTS**: 2,198 high-resolution images with 105,559 pixel-level instance segmentation annotations of apples in MOTS format, captured using UAVs and wearable sensors in an orchard. - https://doi.org/10.5281/zenodo.5939726

## Hyperspectral and Multispectral imaging
- **CitrusFarm Dataset**: CitrusFarm is a multimodal agricultural robotics dataset that provides both multispectral images and navigational sensor data for localization, mapping and crop monitoring tasks - https://ucr-robotics.github.io/Citrus-Farm-Dataset/
- **Multispectral Potato Plants Images**: 360 RGB image patches (750×750 px) with bounding box annotations for object detection tasks in potato crop health assessment - https://datasetninja.com/multispectral-potato-plants-images
- **ARD‑VO (Agricultural Robot Data Set of Vineyards and Olive Groves)** - A real‑world multimodal dataset captured with an unmanned ground vehicle (UGV) navigating vineyards and olive groves in Umbria, Italy, over 11 experimental sessions spanning several kilometers. The dataset includes synchronized sensor data from a stereo-camera rig, a Velodyne LIDAR, a GPS-RTK module with IMU, and a multispectral camera - [Data](https://github.com/isarlab-department-engineering/ARDVO), [Paper](https://doi.org/10.1002/rob.22179)

## Robotics
These datasets support autonomous navigation, localization, and mapping in agriculture and forestry. They usually contain unlabeled multimodal data from a variety of sensors.
- **Sugar Beets 2016**: https://www.ipb.uni-bonn.de/data/sugarbeets2016/
- **CitrusFarm Dataset**: CitrusFarm is a multimodal agricultural robotics dataset that provides both multispectral images and navigational sensor data for localization, mapping and crop monitoring tasks - https://ucr-robotics.github.io/Citrus-Farm-Dataset/
- **ARD‑VO (Agricultural Robot Data Set of Vineyards and Olive Groves)** - A real‑world multimodal dataset captured with an unmanned ground vehicle (UGV) navigating vineyards and olive groves in Umbria, Italy, over 11 experimental sessions spanning several kilometers. The dataset includes synchronized sensor data from a stereo-camera rig, a Velodyne LIDAR, a GPS-RTK module with IMU, and a multispectral camera - [Data](https://github.com/isarlab-department-engineering/ARDVO), [Paper](https://doi.org/10.1002/rob.22179)
- **A high-resolution, multimodal data set for agricultural robotics: A Ladybird's-eye view of Brassica**: https://doi.org/10.1002/rob.21877
- **RELLIS-3D: A Multi-modal Dataset for Off-Road Robotics**: Semantic segmentation on 2D RGB images and **3D LiDAR pointclouds** - https://github.com/unmannedlab/RELLIS-3D/tree/main
- **RUGD Dataset**: The RUGD dataset focuses on semantic understanding of unstructured outdoor environments for applications in off-road autonomous navigation. The datset is comprised of video sequences captured from the camera onboard a mobile robot platform. - http://rugd.vision/
- **GOOSE dataset**: GOOSE is the German Outdoor and Offroad Dataset and is a 2D & 3D semantic segmentation dataset framework. In contrast to existing datasets like Cityscapes or BDD100K, the focus is on unstructured off-road environments - https://goose-dataset.de/docs/
- **WildScenes**: The WildScenes dataset is a multi-modal collection of traversals within Australian forests. The dataset is divided into five sequences across two forest locations. These sequences are both across different physical locations and across different times - https://csiro-robotics.github.io/WildScenes/
- **BotanicGarden**: A robot navigation dataset in a botanic garden of more than 48000m2. Comprehensive sensors are used, including Gray and RGB stereo cameras, spinning and MEMS 3D LiDARs, and low-cost and industrial-grade IMUs. An all-terrain wheeled robot is employed for data collection, traversing through thick woods, riversides, narrow trails, bridges, and grasslands. This yields 33 short and long sequences, forming 17.1km trajectories in total - https://github.com/robot-pesg/BotanicGarden
- **Rosario**: A multi-modal dataset collected in a soybean crop field, comprising over two hours of recorded data from sensors such as stereo infrared camera, color camera, accelerometer, gyroscope, magnetometer, GNSS (Single Point Positioning, Real-Time Kinematic and Post-Processed Kinematic), and wheel odometry. This dataset captures key challenges inherent to robotics in agricultural environments - https://cifasis.github.io/rosariov2/
- **BLT (Bacchus Long Term)**: A multi-session agricultural field dataset collected over months in vineyards (Greece and the UK), featuring seasonal variation, repeated traversal paths and onboard multimodal sensing (RGB-D, LiDAR, navigation). It is designed to support long-term mapping, localisation, crop phenotyping and generalisation studies in robotic agriculture - https://lcas.lincoln.ac.uk/wp/research/data-sets-software/blt/
- **TartanGround** — A large-scale *simulated* dataset for ground robot perception and navigation, comprising 878 trajectories across 63 diverse environments (indoor, natural, rural, urban, industrial, and historical) and 1.44 million samples (~16 TB). Data is collected at 10 Hz using three robot platforms (omnidirectional, differential drive, and quadrupedal legged robot). Each trajectory includes multi-modal sensor data: RGB stereo images (640×640 px, 6-camera 360° rig), depth maps, semantic segmentation, 32-beam LiDAR point clouds, IMU, ground-truth 6-DOF poses, global RGB and semantic point clouds, and ROS bags (legged robot only). Supports tasks such as visual odometry, SLAM, semantic occupancy prediction, and off-road navigation — [Data](https://tartanair.org/tartanground), [Paper](https://arxiv.org/pdf/2505.10696)
- **Crop Row Detection Lincoln Dataset (CRDLD)** — A crop-row detection / semantic segmentation dataset for agricultural robot navigation in maize and sugar beet fields. CRDLD v2.1 contains 2,000 field images (1,250 train / 500 test / 250 val), each paired with ground-truth row labels and MATLAB `.mat` files with labeled coordinates. The dataset spans 50 field-condition classes, including shadows, varying crop growth, weed density, discontinuities, slopes/curves, and tyre tracks. [Data](https://github.com/JunfengGaolab/CropRowDetection), [Paper](https://doi.org/10.1002/rob.22238)

## Collectors of datasets
- **AgML** — An open-source, centralized Python framework for agricultural machine learning. It provides standardized access to public ag-vision datasets (classification, detection, segmentation) — https://github.com/Project-AgML/AgML
- **Dataset Ninja**: https://datasetninja.com/category/agriculture
- **Weed-AI: A repository of Weed Images in Crops**: https://weed-ai.sydney.edu.au/
- **Quantitative Plant**: Website that collects datasets for image classification, semantic segmentation and phenotyping - https://www.quantitative-plant.org/dataset
- **A survey of public datasets for computer vision tasks in precision agriculture**: Collection of datasets for detection and segmentation of weeds and fruits and phenotyping tasks (e.g., damage and disease detection, biomas prediction, yield estimation) - https://doi.org/10.1016/j.compag.2020.105760
- **Weed database development: An updated survey of public weed datasets and cross-season weed detection adaptation**: A survey of 36 publicly available image datasets for weed recognition (classification, detection, and segmentation) + a new two-season dataset of eight weed classes curated for cross-season modeling - https://doi.org/10.1016/j.ecoinf.2024.102546

## Tools to create synthetic datasets
- **CropCraft**: CropCraft is a python script that generates 3D models of crop fields, specialized in real-time simulation of robotics applications - https://github.com/Romea/cropcraft
- **TomatoSynth**: TomatoSynth provides realistic synthetic tomato plants training data for deep learning applications, reducing the need for manual annotation and allowing customization for specific greenhouse environments, thus advancing automation in agriculture - https://github.com/SCT-lab/TomatoSynth
