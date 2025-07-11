# AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems

Autonomous Driving Systems (ADS) are safety-critical, where failures can have severe consequences. Metamorphic Testing (MT) is effective for fault detection in ADS, but existing methods rely heavily on human effort and are hard to automate in industrial pipelines. We present AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that automates the extraction of Metamorphic Relations (MRs) from local traffic rules and generates valid follow-up test cases. AutoMT uses LLMs to extract scenario ontologies and define MRs in Gherkin syntax. A vision-language agent analyzes each scenario, while a search agent retrieves suitable MRs from a RAG-based database to generate follow-up test cases. AutoMT can apply advanced computer vision techniques for MT on real-world datasets or perform MT directly in simulation. Qualitative and quantitative evaluations show that \tool\ generates valid MRs and outperforms baselines by up to 28.26% in generating valid follow-up test and 19.28% in violation detection. AutoMT enables fully automated MT for ADS and can be integrated as a plug-in leveraging local traffic rules and existing test cases in industrial pipelines.


# 1 ADS: Data & Training Pipeline
## 1.1 Data Process 
This project is based on two well-known autonomous driving datasets: A2D2 and Udacity. To process the data, you first need to download these datasets:<br>
Udacity: <a href='https://github.com/udacity/self-driving-car?tab=readme-ov-file' target='_blank'>Self-Driving Car</a><br>
Notice: Due to privacy and data usage restrictions, we cannot provide direct download links for the Udacity dataset. Please obtain it through official channels.<br>
A2D2: <a href='https://www.a2d2.audi/a2d2/en.html' target='_blank'>A2D2</a><br>
For the all dataset, we combine data fromregions, which are then proportionally split into training, validation, and test sets.
After organizing the data, run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/main.py' target='_blank'>main.py</a>.<br> which includes the following steps:<br>
collect_datasets(Type=dataset) - Downsamples images and pairs sensor data<br>
OneFormer.Check_OneFormer(args) - Generates semantic segmentation results<br>
copy_images(Type=dataset) - Resizes images to 320x160 for training<br>
data_process.prepare_data(args) - Loads data into PyTorch structure<br>
train_ADS.Train(args) - Trains the autonomous driving system<br>
trian_ADS.Train(args,dataset,cuda) - Trains the autonomous driving system<br>
        
