# AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems

Autonomous Driving Systems (ADS) are safety-critical, where failures can have severe consequences. Metamorphic Testing (MT) is effective for fault detection in ADS, but existing methods rely heavily on human effort and are hard to automate in industrial pipelines. We present AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that automates the extraction of Metamorphic Relations (MRs) from local traffic rules and generates valid follow-up test cases. AutoMT uses LLMs to extract scenario ontologies and define MRs in Gherkin syntax. A vision-language agent analyzes each scenario, while a search agent retrieves suitable MRs from a RAG-based database to generate follow-up test cases. AutoMT can apply advanced computer vision techniques for MT on real-world datasets or perform MT directly in simulation. Qualitative and quantitative evaluations show that \tool\ generates valid MRs and outperforms baselines by up to 28.26% in generating valid follow-up test and 19.28% in violation detection. AutoMT enables fully automated MT for ADS and can be integrated as a plug-in leveraging local traffic rules and existing test cases in industrial pipelines.
<img src="https://github.com/asvonavnsnvononaon/AutoMT/blob/main/Images/ASE_overall.jpg" width="60%"/>


# 1 ADS: Data & Training Pipeline
## 1.1 Data Prepare 
This project is based on two well-known autonomous driving datasets: A2D2 and Udacity. To process the data, you first need to download these datasets:<br>
Udacity: <a href='https://github.com/udacity/self-driving-car?tab=readme-ov-file' target='_blank'>Self-Driving Car</a><br>
Notice: Due to privacy and data usage restrictions, we cannot provide direct download links for the Udacity dataset. Please obtain it through official channels.<br>
A2D2: <a href='https://www.a2d2.audi/a2d2/en.html' target='_blank'>A2D2</a><br>
The training data for the Autonomous Driving System (ADS) is organized under the Data/ADS_data/ directory. We include two sources of data: Udacity and A2D2, each stored in separate subdirectories for clarity and modularity.<br>
 Udacity Dataset<br>
Directory path: Data/ADS_data/udacity/
Contains the following subfolders:HMB1/,HMB2/,HMB4/,HMB5<br>/,HMB6/<br>
A2D2 Dataset<br>
Directory path: Data/ADS_data/A2D2/<br>
Contains the following sequences:camera_lidar-20180810150607/,camera_lidar-20190401121727/,camera_lidar-20190401145936/<br>
Each subfolder includes camera images and corresponding steering angle and speeds csv provided by the datasets.<br>
For the all dataset, we combine data fromregions, which are then proportionally split into training, validation, and test sets.
## 1.2 Training ADS
After organizing the data, run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/Train_ADS.py' target='_blank'>Train_ADS.py</a>.<br> which includes the following steps:<br>
data_process.data_process(args) - Aligns and synchronizes camera image data with corresponding CAN bus data<br>
collect_datasets(Type=dataset) -  Splits the processed dataset into training, validation, and test sets.<br>
OneFormer.Check_OneFormer(args) - Generates semantic segmentation results<br>
copy_images(Type=dataset) - Resizes images to 320x160 for training<br>
data_process.prepare_data(args) - Loads data into PyTorch structure<br>
train_ADS.Train(args) - Trains the autonomous driving system<br>
trian_ADS.Train(args,dataset,cuda) - Trains the autonomous driving system<br>
<br>
# 2. M-Agent
To enable effective and automatic MR extraction, AUTOMT introduces the M-agent. As illustrated in Figure 2, we use the Gherkin syntax, pre-defined ontology and a LLM agent to define a LLM-based rule parser. A traffic rule is provided to multiple LLM-based rule parsers, which generate candidate MRs. These MRs are then validated using SelfCheckGPT to identify the optimal one. Then all optimal MRs are embedded into a RAG database.<br>
<img src="https://github.com/asvonavnsnvononaon/AutoMT/blob/main/Images/Magent.jpg" width="60%"/><br>
1. run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/M_Agent/M-Agent.py' target='_blank'>M-Agent.py</a>.<br> - extraction MR from Traffic Rules
2. run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/M_Agent/SelfCheckGPT.py' target='_blank'>SelfCheckGPT.py</a>.<br> - Use selfcheckGPT select best MR
3. run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/M_Agent/Generate_RAG.py' target='_blank'>Generate_RAG.py</a>.<br> - Transform MR results become Rag database.

# 3. T-Agent and Rag
## 3.1 AutoMT
1. run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/T_Agent_RAG.py' target='_blank'>T_Agent_RAG.py</a>.<br> -  T-Agent and Rag
2. Install all necessary packages required from this page <a href='https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_csv_rag.ipynb#scrollTo=UBLofB9R27Eg' target='_blank'>csv_rag.py</a>.<br>
## 3.2 Baseline method 
1. run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/Baseline.py' target='_blank'>Baseline.py</a>.<br> -  Baseline
Baseline1: Auto MT Pipeline with LLM-based MR Generation without traffic rule Baseline2: Auto MT Pipeline with LLM-based MR Generation with traffic rule Baseline 3: Auto MT with Manually Generated MR

# Image Generation
1. run <a href='https://github.com/asvonavnsnvononaon/AutoMT/blob/main/Compare_baseline.py' target='_blank'>T_Agent_RAG.py</a>.<br> -  Generation images Set control= 1 or 2
2. control = 1 Generate image from Flux-inpainting and Pix2Pix Install all necessary packages required from this page <br>
<a href='https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev' target='_blank'>Flux inpainting</a>. or Low vram type <a href='https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev' target='_blank'>int4-flux.1-fill-dev</a>.<br>  and <a href='https://github.com/timothybrooks/instruct-pix2pix' target='_blank'>instruct-pix2pix</a>.<br>
3. control = 2 This model adopt the Multimodal LLM to process the reference image and user's editing instruction. 
<a href='https://github.com/stepfun-ai/Step1X-Edit' target='_blank'>Step1X-Edit</a>.<br> Pleas install this model at <a href='https://github.com/asvonavnsnvononaon/AutoMT/tree/main/Other_tools/Step1X-Edit' target='_blank'>Step1X-Edit-file</a>.<br>



# 


   
