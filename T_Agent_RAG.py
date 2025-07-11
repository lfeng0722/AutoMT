import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import os
import pandas as pd
from qwen_vl_utils import process_vision_info
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import json
load_dotenv()
os.environ[
    "OPENAI_API_KEY"] =
# python 3.9
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import base64
import random


def extract_unique_road_types(rules, type=0):
    road_types = set()
    if type == 0:
        pattern = r"Given the ego-vehicle approaches to ([^,]+),"
    else:
        pattern = r"Given the ego-vehicle is ([^,]+),"

    for rule in rules:
        match = re.search(pattern, rule)
        if match:
            #road_types.add(match.group(1).strip())
            road_types.add(match.group(1).strip().lower())
    return sorted(road_types)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

global AutoMT_German, AutoMT_California,Num




##########################




####################

class Matching_MR:
    def __init__(self, RAG_location="German", VLM="Qwen"):
        # Two cuda  device
        cuda_type = 1
        if cuda_type == 2:
            self.cuda1 = "cuda:0"
            self.cuda2 = "cuda:1"
        else:
            self.cuda1 = "cuda"
            self.cuda2 = "cuda"
        self.VLM = VLM
        if self.VLM == "Qwen":
            self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="auto",
                device_map=self.cuda1
            )
        else:
            self.client = OpenAI(
                api_key=)
        if RAG_location == "German":
            self.road_types = extract_unique_road_types(AutoMT_German)
            self.MRS = AutoMT_German
        else:
            self.road_types = extract_unique_road_types(AutoMT_California)
            self.MRS = AutoMT_California
        self.vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.RAG_location = RAG_location
        self.LLMmodel = "gpt-4.1"  #  -mini-2025-04-14
        self.llm = ChatOpenAI(model=self.LLMmodel)
        self.update_RAG()
        self.rag_chain = self.load_RAG()
        self.selected_mrs = []  # Store previously selected MRs
        self.max_history = 3  # Maximum number of MRs to keep in history
        self.Wrong = 0
        self.data = []

    def load_RAG(self):
        loader = CSVLoader(file_path=self.file_path)
        docs = loader.load_and_split()
        embeddings = OpenAIEmbeddings()
        index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
        vector_store = FAISS(
            embedding_function=OpenAIEmbeddings(),
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(documents=docs)

        retriever = vector_store.as_retriever(search_kwargs={"k": len(self.MRS)})

        # Set up system prompt
        system_prompt = (

            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),

        ])

        # Create the question-answer chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain

    def analyze_media(self, images):
        if self.VLM == "Qwen":
           # prompt = (
           #     f"Analyze this driving scene. Describe the road types (must be one of the following: {', '.join(self.road_types)}), "
           #     f"'any roads' can be used if no specific type fits. "
          #      f"Reply format: road network: <road_type>"
            #)
            #prompt = (f"Analyze this driving scene. Describe the time of day, weather conditions, road type (Only one of these {', '.join(self.road_types)}.) 'any roads' can be used only when none of the specific road types fit.  and any objects around the vehicle. Reply format: time: [2 words], weather: [2 words], road: [1-2 words], objects: ")
            prompt = (f"Analyze this driving scene. Describe the road type (must chosen from one of these {', '.join(self.road_types)}.) If there no suitable one, 'any roads' can be used only when none of the specific road types fit.  output: road: [1-2 words]  ")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": images,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.vl_model.device)
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text
        else:
            base64_image = encode_image(images[0])
            response = self.client.responses.create(
                model=self.LLMmodel,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",
                             "text": f"Analyze this driving scene. Describe the time of day, weather conditions, road type (such as {', '.join(self.road_types)}.) 'any roads' can be used only when none of the specific road types fit.  and any objects around the vehicle. Reply format: time: [2 words], weather: [2 words], road: [1-2 words], objects: "},
                            {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                        ],
                    }
                ],
            )
            return response.output[0].content[0].text

    def match_mr_with_llm(self, scene_description, infos, chose=1):
        result = ", ".join([f"index: {i}, weight of index: {val}" for i, val in enumerate(self.usage)])
        print(scene_description)
        Scen  = re.search(r'road:\s*(\w+)', scene_description[0]).group(1)
        if Scen not in self.road_types:
            Scen = "any roads"
        chose = 0
        if chose == 1:

            if self.VLM=="1":
                scene_description = scene_description[0].split(":", 1)[1].strip()
                candidate_index = int(random.choice(self.total_mrs))
                print(candidate_index)
                match = re.search(r"Given the ego-vehicle approaches to ([^,]+),", self.MRS[candidate_index])

                prompt = f"""
                You are provided with:
                1. A driving scene description.
                2. A candidate metamorphic relation (MR) with its index.
                3. Several other MRs retrieved from context.
    
                **Task**:
                Scene: "{scene_description}"
                - First, identify the **Road Network type** from the candidate MR at index {candidate_index}.
                This Road Network of MR is {match}
                - Then, determine whether this MR is **semantically suitable** for the scene based on **similarity of Road Network** (not necessarily exact match).
                - "Similar" means the road structure or traffic layout shares common semantics. For example, "any road" is similar to "All Roads".
                If the candidate MR's Road Network is **similar** to the one described in the scene, return: `index={candidate_index}`.
                vehicle info: {infos}
                **Decision logic**:
                - Otherwise, randomly select an alternative MR **from the context** that matches the **Road Network type or structure**, and return: `index=XX`.
                - If no MR seems clearly similar, return a random index from the context.
    
    
    
                **Return format** (only one line):  
                index=XX
                """

            else:
                scene_description = scene_description[0].split(":", 1)[1].strip()

                prompt = f"""
                You are given:

                - A current road type: {scene_description}
                - Ego-vehicle information: {infos}
                - A list of weights for MRs: {result}
                - A set of MRs retrieved via RAG. Each MR contains:
                  - road type
                  - modification action
                  - expected behavior
                
                Your task is to:
                1. Identify all MR indexes whose road type is exactly or semantically equivalent to the current road type.
                2. From these, keep only those that are logically applicable to the ego-vehicle state (e.g., speed too high for “slow down”).
                3. Among the applicable MRs, select the one with the lowest weight.
                Return as:
                First, I identify all MRs whose road type exactly matches to the current road type as [index: i, index k...]
                Among these candidates, Then, I sort these candidate MRs in ascending order of weight (i.e., from lowest to highest). Sort=[[index: i, weight of index: j]...]
                Starting from the index in Sort with the lowest weight is A.
                I examine whether it is logically applicable to the current ego-vehicle state result is yes or no.
                If it is logically applicable, I select this MR.
                Double check the road type of selected MR. If the road type !=   {scene_description},I move on to the MR with the next lowest weight and repeat the check.
                If it is not logically applicable, I move on to the MR with the next lowest weight and repeat the check.
                If none of the MRs are logically applicable, I fall back to the Sort with the lowest weight among the road-type-matched candidates.
                Thus, the final selected MR is:
                indexes=[i]
                """
            answer = self.rag_chain.invoke({"input": prompt})

            match = re.search(r'indexes\s*=\s*\[([0-9,\s]+)\]', answer["answer"])


            if match:
                index_str = match.group(1)
                try:
                    matched_indexes = [int(i.strip()) for i in index_str.split(',')]
                    # 过滤非法 index（只保留在 total_mrs 中的）
                    matched_indexes = [i for i in matched_indexes if i in self.total_mrs]
                    # 如果全部非法，随机给一个合法的
                    if not matched_indexes:
                        matched_indexes = [random.choice(self.total_mrs)]
                        self.Wrong += 1
                except:
                    matched_indexes = [random.choice(self.total_mrs)]
                    self.Wrong += 1

                index_value = min(matched_indexes, key=lambda i: self.usage[i])
            else:
                index_value = random.choice(self.total_mrs)
                self.Wrong += 1

            # 更新使用次数
            self.usage[index_value] += 1
            #self.update_RAG(index=index_value)
           #self.rag_chain = self.load_RAG()
            pattern = r"Given the ego-vehicle approaches to ([^,]+),"
            match = re.search(pattern, self.MRS[index_value])

            print("VLM selected road type:",scene_description,"||||", "RAG matched road type:", match.group(1).strip(),"||||",index_value)
            print(self.usage)

            log_data = {
                "road type": scene_description,
                "vehicle_info": infos,
                "extracted_road_type":  match.group(1).strip(),
                "llm_answer":  answer["answer"],
                "final_index_selected": index_value,
                "MR": self.MRS[index_value]
            }
            self.data.append(log_data)
            if self.RAG_location=="German":
                out = "German.json"
            else:
                out = "output.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return index_value

        else:
            matched_indices = [i for i, mr in enumerate(self.MRS) if Scen in mr]

            # Step 3: 根据匹配情况随机选择 index
            if matched_indices:
                selected_index = random.choice(matched_indices)
            else:
                selected_index = random.randint(0, len(self.MRS) - 1)
            pattern = r"Given the ego-vehicle approaches to ([^,]+),"
            match = re.search(pattern, self.MRS[selected_index])
            print("VLM selected road type:", scene_description, "||||", "RAG matched road type:",
                  match.group(1).strip(), "||||", selected_index)
            self.usage[selected_index] += 1
            print(self.usage)
            return selected_index

    def find_wrong(self):
        return self.Wrong

    def update_RAG(self, index=None):
        if index is None:
            if self.RAG_location == "German":
                self.file_path = ("MR_generate/"+str(Num)+"/"+"RAG_German.csv")
            elif self.RAG_location == "California":
                self.file_path = ("MR_generate/"+str(Num)+"/"+"RAG_California.csv")
            original_df = pd.read_csv(self.file_path)
            original_df['weight'] = 0
            self.total_mrs = original_df['index'].tolist()

            original_df.to_csv("MR_generate/temp.csv", index=False)
            self.usage = [0] * len(self.MRS)


        else:
            self.file_path = "MR_generate/temp.csv"
            weights_df = pd.read_csv("MR_generate/temp.csv")
            idx_pos = weights_df['index'] == index
            if any(idx_pos):
                weights_df.loc[idx_pos, 'weight'] += 1
            result_df = weights_df.sample(frac=1)

            result_df.to_csv(self.file_path, index=False)

    def match_MR(self, images):
        output = self.analyze_media(images)
        match_result = self.match_mr_with_llm(output)
        return match_result


def separate_image_paths():
    excel = os.path.join("Data", "test_dataset.xlsx")
    matched_data = pd.read_excel(excel, sheet_name="Sheet1")

    # 提取字段
    image_files = matched_data["Image File"]
    steering_angles = matched_data["Steering Angle"]
    vehicle_speeds = matched_data["Vehicle Speed"]

    # 定义前缀
    udacity_prefix = "Data/ADS_data/udacity"
    a2d2_prefix = "Data/Raw/A2D2"

    udacity_paths = []
    a2d2_paths = []
    udacity_infos = []
    a2d2_infos = []

    # 遍历所有图像路径
    for i, path in enumerate(image_files):
        angle = steering_angles[i]
        speed_kph = vehicle_speeds[i] * 3.6  # 转换为 km/h
        info = f"Ego vehicle steering Angle={angle:.2f} rad, Ego-vehicle Speed={speed_kph:.2f} km/h"
        if path.startswith(udacity_prefix):
            udacity_paths.append(path)
            udacity_infos.append(info)
        elif path.startswith(a2d2_prefix):
            a2d2_paths.append(path)
            a2d2_infos.append(info)

    return udacity_paths, udacity_infos, a2d2_paths, a2d2_infos


import csv
import json


def process_path(paths, infos, RAG_location, type=1):
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    mr_LLM = Matching_MR(RAG_location=RAG_location)

    video_length = 10
    results = []
    total_paths = len(paths)

    iiii = 0
    for i in range(0, total_paths, video_length):
        #time.sleep(1)
        video_frames = []
        # 获取当前批次的图片路径（最多10张）
        batch_paths = paths[i:min(i + video_length, total_paths)]

        for path in batch_paths:
            local_path = os.path.join("C:/Users/Administrator/Desktop/ITMT_EXP/Data/test", os.path.basename(path))
            video_frames.append(local_path)

        if type == 1:

            analysis_txt = mr_LLM.analyze_media(video_frames)

            matched_txt = mr_LLM.match_mr_with_llm(analysis_txt, infos[i + 5])
        else:
            matched_txt = mr_LLM.match_mr_with_llm(analysis_txt, infos[i + 5], chose=2)

        results.append({
            "frames": video_frames[0],
            "analysis": analysis_txt,
            "matched_result": matched_txt,
            "Wrong_number": mr_LLM.find_wrong()
        })

        with open("temp.csv", 'w', encoding='utf-8', newline='') as f:
            fieldnames = ["frames", "analysis", "matched_result", "Wrong_number"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return results


def save_to_csv(results, output_file):
    """首次写入（例如保存 Udacity）"""
    fieldnames = ["frames", "analysis", "matched_result", "Wrong_number"]
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def append_to_csv(results, output_file):
    """追加写入（例如保存 A2D2）"""
    fieldnames = ["frames", "analysis", "matched_result", "Wrong_number"]
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for result in results:
            writer.writerow(result)
import pandas as pd
import re

if __name__ == "__main__":
    udacity_paths, udacity_infos, a2d2_paths, a2d2_infos = separate_image_paths()  # Assuming this function is defined elsewhere
    """
    california_results = process_path(udacity_paths, udacity_infos, "California_1",type=2)
    save_to_csv(california_results, "combined_analysis_results_4.csv")
    german_results = process_path(a2d2_paths,a2d2_infos, "German_1",type=2)
    append_to_csv(german_results, "combined_analysis_results_4.csv")
    print("_________________Success____________________")
    california_results = process_path(udacity_paths, udacity_infos, "California", type=2)
    save_to_csv(california_results, "combined_analysis_results_3.csv")
    german_results = process_path(a2d2_paths,a2d2_infos, "German", type=2)
    append_to_csv(german_results, "combined_analysis_results_3.csv")
    print("_________________Success____________________")
    """
    for Num in range(5):


        df = pd.read_csv("MR_generate/"+str(Num)+"/"+"lowest_score_models.csv", encoding="latin1")  # ISO-8859-1 兼容大部分字符
        Best_MR = df["Best_MR"]
        AutoMT_German = Best_MR[0:38]
        AutoMT_California = Best_MR[38:110]
        AutoMT_German = AutoMT_German.tolist()
        AutoMT_California = AutoMT_California.tolist()

        california_results = process_path(udacity_paths, udacity_infos, "California")
        save_to_csv(california_results, "AutoMT" + str(Num) + ".csv")
        german_results = process_path(a2d2_paths, a2d2_infos, "German")
        append_to_csv(german_results, "AutoMT" + str(Num) + ".csv")
    #  print("_________________Success____________________")
    # california_results = process_path(udacity_paths, udacity_infos, "California_1")
    # save_to_csv(california_results, "combined_analysis_results_2.csv")
    # german_results = process_path(a2d2_paths,a2d2_infos, "German_1")
    # append_to_csv(german_results, "combined_analysis_results_2.csv")
    print("_________________Success____________________")


#[1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 22, 10, 10, 10, 10, 10, 0, 9, 0, 0, 0, 0, 0, 11, 0, 0, 10, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 10, 9, 9, 0, 0, 0, 0, 9, 0, 0]
#[9, 9, 9, 8, 0, 0, 6, 6, 6, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 38, 6, 8, 8, 8, 8, 8, 0, 6, 6, 5, 5, 14, 8]

#[2, 1, 5, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 7, 8, 2, 3, 5, 0, 3, 3, 7, 1, 0, 0, 2, 9, 0, 0, 1, 0, 2, 0, 0, 1, 10, 2, 5, 0, 0, 0, 0, 0, 0, 2, 0, 7, 3, 5, 3, 4, 2, 4, 6, 0, 0, 4, 2, 3, 0, 3, 7, 3, 1, 8, 3]
#[7, 11, 2, 14, 10, 0, 4, 6, 8, 7, 5, 9, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 11, 4, 3, 6, 3, 12, 13, 12, 12, 7, 9, 7, 9, 7, 3, 10]

#[6, 5, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 5, 6, 2, 2, 5, 0, 3, 2, 4, 0, 1, 4, 0, 4, 0, 0, 5, 0, 0, 1, 3, 1, 0, 7, 3, 0, 0, 1, 0, 0, 0, 4, 0, 3, 4, 1, 9, 6, 6, 4, 9, 0, 0, 3, 3, 2, 3, 3, 4, 5, 8, 5, 3]
#[9, 7, 11, 7, 8, 0, 7, 10, 6, 16, 6, 9, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 9, 5, 7, 8, 8, 5, 10, 0, 5, 8, 11, 6, 13, 8, 5, 4]

#[3, 3, 3, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 6, 6, 4, 3, 3, 1, 0, 6, 2, 4, 1, 1, 1, 1, 5, 0, 0, 4, 0, 1, 0, 1, 0, 5, 4, 5, 0, 0, 0, 0, 0, 0, 5, 0, 5, 6, 8, 4, 5, 1, 4, 4, 0, 0, 6, 7, 4, 5, 5, 5, 2, 1, 6, 0]
#[9, 6, 11, 3, 5, 0, 13, 6, 10, 3, 7, 6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 7, 15, 4, 11, 7, 11, 12, 14, 5, 8, 4, 8, 9, 7, 10]

#[3, 1, 9, 0, 11, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 2, 6, 4, 3, 5, 0, 3, 5, 6, 0, 1, 4, 1, 1, 0, 0, 4, 0, 0, 0, 1, 1, 5, 6, 5, 0, 0, 4, 0, 0, 0, 2, 0, 5, 4, 4, 0, 3, 3, 6, 2, 0, 0, 7, 2, 7, 1, 4, 5, 4, 1, 4, 4]
#[6, 10, 11, 6, 10, 0, 8, 9, 7, 5, 10, 13, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 6, 9, 6, 10, 8, 8, 0, 9, 7, 10, 5, 2, 7, 10, 8, 9]



