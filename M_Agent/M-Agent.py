System_prompt = """You are an expert in traffic rules and scene analysis. Metamorphic Testing (MT) is a method used in autonomous vehicle testing. Your task is to convert traffic rules into structured "Given-When-Then" metamorphic relations (MRs) for vehicle testing. 
# Key Concepts #
1. traffic rule: Define how the ego-vehicle should behavioral in the specific driving scenario.
2. Road Typ: Road elements are specified in the traffic rule, such as crosswalk.
3. Manipulation: adds objects specified in the traffic rule, such as  red light, or replaces environmental conditions, such as a rainy day.
4. Ego-Vehicle Expected Behavior: The expected ego-vehicle behavior in the traffic rule, such as slow down, turn right.
# EXAMPLE #
User: Traffic rule: "Steady Red Light (Stop) Stop before entering the crosswalk or intersection"  
Assistant: Given the ego-vehicle approaches to an intersection, When AUTOMT adds a steady red light on the roadside,Then ego-vehicle should slow down"""


def get_prompt(rule, selected):
    Road_type = ["crosswalk", "bus stopping", "field path", "any roads", "school area", "port area", "industrial area",
                 "intersection", "railroad crossings", "alleys"]

    Manipulation = ["st. andrew's cross sign", "bus stop sign", "Red and White Regulatory Sign",
                    "start of 30 km/h zone sign", "maximum speed limit sign", "prohibition for motor vehicles sign",
                    "prohibition for vehicles of all kinds sign", "warning signs", "right-of-way regulation signs",
                    "stop sign", "yield sign", "5-sided sign", "red light", "yellow light", "green light", "red arrow",
                    "green arrow", "crosswalk markings", "limit lines", "stopped vehicle", "oncoming vehicle",
                    "oncoming bicycle", "oncoming bicyclist", "oncoming vehicle", "turning vehicle", "rail vehicle",
                    "public transport bus", "school bus", "line bus", "motor vehicle", "multi-track motor vehicle",
                    "emergency vehicle", "tow truck", "road work vehicle", "pedestrian", "person using roller skates",
                    "person using a skateboard", "person with a disability using a wheelchair",
                    "person with a disability using a tricycle", "person with a disability using a quadricycle",
                    "child", "senior (elderly person)", "person with small children", "bicyclist", "bicycle",
                    "bicycle with auxiliary motor", "electric micro-vehicle", "passenger", "driver in front",
                    "vehicle behind", "railway employee with flag", "obstacle on the road", "road narrowing",
                    "heavy traffic", "rain environment", "snow environment", "mud environment", "ice road", "wet road",
                    "fog environment", "heavy smoke environment", "high winds environment", "low lighting environment"]

    Expected_Behavior = ["slow down", "turn left", "turn right", "keep current"]

    prompt = f"""
    You are given:
    1. Detalies of the domain-specific language (DSL) of MRs:  of Road Type:{Road_type}, Manipulation:{Manipulation} and Ego-Vehicle Expected Behavior:{Expected_Behavior}.
    Notice: If the traffic rule does not specify a particular road type, analyze whether the rule is generally applicable across all types of roads.
    2. User: {rule}
    3. You only choose elements clearly mentioned A from Road Type ontology elements. 
    You only choose elements clearly mentioned B from Manipulation ontology elements. 
    You only choose elements clearly mentioned C from go-Vehicle Expected Behavior. 
    DO NOT guess or hallucinate extra elements.
    4. For Manipulation:
    - Only choose elements clearly mentioned or implied in the rule.
    Answer as:
    - I first determine one appropriate Road Type ontology element based on the rule. This Road Type is: A  
    - Is A in the Road Type ontology list? Answer: Yes/No  
    - If No, re-select a valid Road Type from the ontology that matches the rule. The corrected Road Type is: A'

- Then I determine All appropriate Manipulation ontology elements based on the rule. This Manipulation ontology elements are: ...
    Is there any Manipulation in Manipulation ontology list and is not selected Manipulation?
    If yes, The corrected Manipulation is: B.
    If No, select one Manipulation ontology from ontology list that matches the rule.The corrected Manipulation is: B

    If multiple Manipulation elements are found, select only one and do NOT list multiple Manipulations.



- I determine the verb for Manipulation, use adds for optional objects (e.g., pedestrians, vehicles), and replaces for objects with mandatory presence (e.g., weather, lighting conditions). This verb is: C  
    - Is C either 'adds' or 'replaces'? Answer: Yes/No  
    - If No, select either 'adds' or 'replaces' based on the nature of B. The corrected verb is: C'

- I determine one appropriate Ego-Vehicle Expected Behavior ontology element based on the rule. This Ego-Vehicle Expected Behavior is: D  
    - Is D in the Ego-Vehicle Expected Behavior ontology list? Answer: Yes/No  
    - If No, re-select a valid behavior from the ontology that matches the rule. The corrected behavior is: D

    Given a manipulation element B and its operation type C ("adds" or "replaces"), generate an enhanced version B' that includes only **one minimal and necessary layer of detail**, based on the type of manipulation:

    - If C is "adds", B' must only specify the added object with a minimal qualifier (e.g., object status or general presence), such as:  
      → B: adds red light  
      → B': adds a steady red light on the roadside  
       Keep it generic and minimal  
       Do NOT include extra details like specific locations, times, weather, scene context, or any descriptive elaboration.

    - If C is "replaces", B' must clearly describe only the replacement action in the form:  
      "replaces [original element] into [new element]", such as:  
      → B: replaces rainy day  
      → B': replaces the weather into rainy day  
     Do NOT elaborate with extra descriptors, conditions, or scenario details.


    Finally, compose the MR using the selected (or corrected) elements in the following format:  
["Given the ego-vehicle approaches to A, When AUTOMT C B', Then ego-vehicle should D"]
    """
    return prompt


import re


def Claude(client, System_prompt, prompt, select):
    prompt = get_prompt(prompt, select)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        temperature=0.1,
        system=System_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    response = message.content[0].text

    match = re.search(r'\["(Given the ego-vehicle approaches to .*?Then ego-vehicle should.*?)"\]', response)
    if match:
        mr = match.group(1)
    else:
        mr = "not match"

    return mr


# -mini-2025-04-14
def ChatGPT(client, System_prompt, prompt, select):
    prompt = get_prompt(prompt, select)
    response = client.responses.create(
        model="gpt-4.1",  # -mini-2025-04-14
        input=[
            {
                "role": "developer",
                "content": System_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1  # 控制生成文本的随机性
    )
    response = response.output_text
    match = re.search(r'\["(Given the ego-vehicle approaches to .*?Then ego-vehicle should.*?)"\]', response)
    if match:
        mr = match.group(1)
    else:
        mr = "not match"

    return mr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 清除缓存
torch.cuda.empty_cache()
#del chatbot


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,#"auto",
            device_map={"": "cuda:0"}
        )
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=2048)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        match = re.search(r'\["(Given the ego-vehicle approaches to .*?Then ego-vehicle should.*?)"\]', response)
        if match:
            mr = match.group(1)
        else:
            mr = "not match"
        return mr
chatbot = QwenChatbot()

import time
import anthropic
from openai import OpenAI

GTP_API =
Claude_API =
Claude_client = anthropic.Anthropic(api_key=Claude_API)
ChatGPT_client = OpenAI(api_key=GTP_API)
for i in range(4):
    folder_path = str(i)
    os.makedirs(folder_path, exist_ok=True)

    all_rules_df = pd.DataFrame(columns=["Region", "Traffic Rule", "Metamorphic Relation"])
    for region in sheet_names:
        sheet_data = pd.read_excel('Traffic Rules.xlsx', sheet_name=region)
        traffic_rules = sheet_data['Traffic Rules'].tolist()
        select = None
        for rule in traffic_rules:
            mr = ChatGPT(ChatGPT_client, System_prompt, rule, select)
            match = re.search(r"When AUTOMT (replaces|adds) the ([^,]+),", mr)
            if match:
                verb = match.group(1)  # "replaces" 或 "adds"
                select = match.group(2).strip()  # 介于 'the' 和 ',' 之间的内容

            # print("Traffic rule: ",rule)
            # print("MR: ",mr)

            new_row = pd.DataFrame({
                "Region": [region],
                "Traffic Rule": [rule],
                "Metamorphic Relation": [mr]
            })
            all_rules_df = pd.concat([all_rules_df, new_row], ignore_index=True)
            all_rules_df.to_excel(folder_path + "/" + "GPT_Traffic_Rules_Results.xlsx", index=False, engine='openpyxl')
    all_rules_df.to_excel(folder_path + "/" + "GPT_Traffic_Rules_Results.xlsx", index=False, engine='openpyxl')
import os

for i in range(1, 4):
    folder_path = str(i)
    os.makedirs(folder_path, exist_ok=True)
    all_rules_df = pd.DataFrame(columns=["Region", "Traffic Rule", "Metamorphic Relation"])
    for region in sheet_names:
        sheet_data = pd.read_excel('Traffic Rules.xlsx', sheet_name=region)
        traffic_rules = sheet_data['Traffic Rules'].tolist()
        select = []
        for rule in traffic_rules:
            mr = Claude(Claude_client, System_prompt, rule, select)
            match = re.search(r"Given the ego-vehicle approaches to ([^,]+),", mr)
            if match:
                A = match.group(1)
                select.append(A)

            print("Traffic rule: ", rule)
            print("MR: ", mr)
            new_row = pd.DataFrame({
                "Region": [region],
                "Traffic Rule": [rule],
                "Metamorphic Relation": [mr]
            })
            all_rules_df = pd.concat([all_rules_df, new_row], ignore_index=True)
            all_rules_df.to_excel(folder_path + "/" + "Claude_Traffic_Rules_Results.xlsx", index=False,
                                  engine='openpyxl')
    all_rules_df.to_excel(folder_path + "/" + "Claude_Traffic_Rules_Results.xlsx", index=False, engine='openpyxl')

import re
from tqdm import tqdm


def get_new_prompt(rule):
    Road_type = ["crosswalk", "bus stopping", "field path", "any roads", "school area", "port area", "industrial area",
                 "intersection", "railroad crossings", "alleys"]

    Manipulation = ["st. andrew's cross sign", "bus stop sign", "start of 30 km/h zone sign",
                    "maximum speed limit sign", "no entry sign", "prohibition for motor vehicles sign",
                    "prohibition for vehicles of all kinds sign", "warning signs", "right-of-way regulation signs",
                    "stop sign", "yield sign", "5-sided sign", "red light", "yellow light", "green light", "red arrow",
                    "green arrow", "crosswalk markings", "limit lines", "stopped vehicle", "oncoming vehicle",
                    "oncoming bicycle", "oncoming bicyclist", "oncoming vehicle", "turning vehicle", "rail vehicle",
                    "public transport bus", "school bus", "line bus", "motor vehicle", "multi-track motor vehicle",
                    "emergency vehicle", "tow truck", "road work vehicle", "pedestrian", "person using roller skates",
                    "person using a skateboard", "person with a disability using a wheelchair",
                    "person with a disability using a tricycle", "person with a disability using a quadricycle",
                    "child", "senior (elderly person)", "person with small children", "bicyclist", "bicycle",
                    "bicycle with auxiliary motor", "electric micro-vehicle", "passenger", "driver in front",
                    "vehicle behind", "railway employee with flag", "obstacle on the road", "road narrowing",
                    "heavy traffic", "rain environment", "snow environment", "mud environment", "ice road", "wet road",
                    "fog environment", "heavy smoke environment", "high winds environment", "low lighting environment"]

    Expected_Behavior = ["slow down", "turn left", "turn right", "keep current"]

    prompt = f"""
    You are given:
    1. Detalies of the domain-specific language (DSL) of MRs:  of Road Type:{Road_type}, Manipulation:{Manipulation} and Ego-Vehicle Expected Behavior:{Expected_Behavior}.
    Notice: If the traffic rule does not specify a particular road type, analyze whether the rule is generally applicable across all types of roads.
    2. User: {rule}
    3. You only choose elements clearly mentioned A from Road Type ontology elements. 
    You only choose elements clearly mentioned B from Manipulation ontology elements. 
    You only choose elements clearly mentioned C from go-Vehicle Expected Behavior. 
    DO NOT guess or hallucinate extra elements.
    4. For Manipulation:
    - Only choose elements clearly mentioned or implied in the rule.
    Answer as:
    - I first determine one appropriate Road Type ontology element based on the rule. This Road Type is: A  
    - Is A in the Road Type ontology list? Answer: Yes/No  
    - If No, re-select a valid Road Type from the ontology that matches the rule. The corrected Road Type is: A'

- Then I determine All appropriate Manipulation ontology elements based on the rule. This Manipulation ontology elements are: ...
    Is there any Manipulation in Manipulation ontology list and is not selected Manipulation?
    If yes, The corrected Manipulation is: B.
    If No, select one Manipulation ontology from ontology list that matches the rule.The corrected Manipulation is: B

    If multiple Manipulation elements are found, select only one and do NOT list multiple Manipulations.



- I determine the verb for Manipulation, use adds for optional objects (e.g., pedestrians, vehicles), and replaces for objects with mandatory presence (e.g., weather, lighting conditions). This verb is: C  
    - Is C either 'adds' or 'replaces'? Answer: Yes/No  
    - If No, select either 'adds' or 'replaces' based on the nature of B. The corrected verb is: C'

- I determine one appropriate Ego-Vehicle Expected Behavior ontology element based on the rule. This Ego-Vehicle Expected Behavior is: D  
    - Is D in the Ego-Vehicle Expected Behavior ontology list? Answer: Yes/No  
    - If No, re-select a valid behavior from the ontology that matches the rule. The corrected behavior is: D'

    B' is the Manipulation element B enhanced with necessary details:
    - If C is "adds", B' includes the location where the object is added (e.g., C B: adds red light ->B': adds a steady red light on the roadside).  
    - If C is "replaces", B' includes the replaced object and its replacement (e.g., C B replaces rainy day -> replaces the weather into rainy day).


    Finally, compose the MR using the selected (or corrected) elements in the following format:  
["Given the ego-vehicle approaches to A, When AUTOMT C B', Then ego-vehicle should D./no_think"]
    """
    return prompt


import os

for i in range(4):
    folder_path = str(i)
    os.makedirs(folder_path, exist_ok=True)
    all_rules_df = pd.DataFrame(columns=["Region", "Traffic Rule", "Metamorphic Relation"])
    for region in sheet_names:
        sheet_data = pd.read_excel('Traffic Rules.xlsx', sheet_name=region)
        traffic_rules = sheet_data['Traffic Rules'].tolist()
        select = []
        for rule in tqdm(traffic_rules, desc="Judging traffic rules"):
            promt = get_new_prompt(rule)
            mr = chatbot.generate_response(promt)
            # print("Traffic rule: ",rule)
            # print("MR: ",mr)
            new_row = pd.DataFrame({
                "Region": [region],
                "Traffic Rule": [rule],
                "Metamorphic Relation": [mr]
            })
            all_rules_df = pd.concat([all_rules_df, new_row], ignore_index=True)
            all_rules_df.to_excel(folder_path + "/" + "Qwen_Traffic_Rules_Results.xlsx", index=False, engine='openpyxl')
    all_rules_df.to_excel(folder_path + "/" + "Qwen_Traffic_Rules_Results.xlsx", index=False, engine='openpyxl')




