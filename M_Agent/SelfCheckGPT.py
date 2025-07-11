#Two Types
#1.Juage and Select
#2.Auto select from LLM
#Type 1

System_prompt = """# Based on the list of close-ended yes or no questions, generate a JSON answer.
# TASK # Given a traffic rule, convert it into a structured "Given-When-Then" metamorphic relation (MR) for vehicle testing.
# KEY CONCEPTS #
[VEHICLE MANEUVER]: The expected ego-vehicle maneuver in the traffic rule, such as slow down, turn right.
[ROAD NETWORK]: Road elements are specified in the traffic rule, such as lines and crosswalk.
[MODIFICATION TARGET]: One object or environment is specified in the traffic rule, such as school zone, red light, rainy day

# OBJECTIVE # Provide yes/no answers to the given questions based on the provided MR and traffic rule.
The elements of MR are: MR template.
"Given the ego-vehicle approaches to |[ROAD NETWORK]|,
When ITMI |[adds/replaces]| |[MODIFICATION TARGET]|,
Then ego-vehicle should |[VEHICLE MANEUVER]|"

# STYLE # Generate a JSON object with answers for all questions in the following format:
IMPORTANT: 
- Only return the JSON object, nothing else.
- The 'answers' key must contain a list of strings, either 'yes' or 'no'.
- The number of answers must exactly match the number of questions.
- Answer 'no' if there's not enough information to answer confidently.

Questions:
1. Are [ROAD NETWORK], [MODIFICATION TARGET], and [VEHICLE MANEUVER] all mentioned in the traffic rule?
2. Is the traffic rule supported by MR?
3. Are all parts of the MR consistent with each other?
4. Does the generated result reflect a realistic scenario where the [VEHICLE MANEUVER] would actually occur?
"""


def LLama_hallucination_detection(client, System_prompt, rule, MR):
    Example_Q = """
    Traffic rule: Steady Red Light (Stop) Stop before entering the crosswalk or intersection.
    MR: Given the ego-vehicle approaches to |an intersection|,
When ITMI |adds| |a steady red light on the roadside|,
Then ego-vehicle should |slow down|
    """
    Example_A = """["yes", "yes", "yes", "yes"]"""

    # Combine the examples and the new query into a single prompt
    full_prompt = f"""{Example_Q}

{Example_A}

Traffic rule: {rule}
MR: {MR}"""

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "developer",
                "content": System_prompt
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        temperature=0.1  # Controls randomness in generation
    )

    return response.output_text


def parse_json(json_str):
    # 使用正则匹配真正的 JSON 部分
    match = re.search(r'```json\s*({.*?})\s*```', json_str, re.DOTALL)
    if not match:
        return None  # 没有匹配到 JSON

    try:
        parsed = json.loads(match.group(1))
        return parsed
    except json.JSONDecodeError:
        return None


def calculate_score(answers):
    score = 0
    total_answers = 0
    for question, answer_list in answers.items():
        total_answers += len(answer_list)
        for answer in answer_list:
            if answer.lower() == 'no':
                score += 1
    return score / total_answers if total_answers > 0 else 0


models =["GPT","Claude","Qwen"]
import pandas as pd
sheet_data = pd.read_excel(f'{models[0]}_Traffic_Rules_Results.xlsx', sheet_name="Sheet1")
Claude_results = sheet_data['Metamorphic Relation'].tolist()
sheet_data = pd.read_excel(f'{models[1]}_Traffic_Rules_Results.xlsx', sheet_name="Sheet1")
GPT_results = sheet_data['Metamorphic Relation'].tolist()
sheet_data = pd.read_excel(f'{models[2]}_Traffic_Rules_Results.xlsx', sheet_name="Sheet1")
LLama_results = sheet_data['Metamorphic Relation'].tolist()

import pandas as pd
import re
import json
def get_score_for_mr(ChatGPT_client,system_prompt,rule,mr):
    response=LLama_hallucination_detection(ChatGPT_client,system_prompt,rule,mr)
    parsed=parse_json(response)
    if parsed:return calculate_score(parsed)
    return 0
models=["GPT","Claude","Qwen"]
all_data={}
for model in models:
    df = pd.read_excel(f"{model}_Traffic_Rules_Results.xlsx")  # ✅ 使用 read_excel 代替 read_csv
    all_data[model] = {
        "rules": df["Traffic Rule"],
        "mrs": df["Metamorphic Relation"]
    }

results=pd.DataFrame(columns=["Rule","Lowest_Model","Lowest_Score","Best_MR"])

for i in range(len(all_data[models[0]]["rules"])):
    rule = all_data[models[0]]["rules"][i]
    scores = {}
    mrs = {}
    for model in models:
        mr = all_data[model]["mrs"][i]
        mrs[model] = mr
        scores[model] = get_score_for_mr(ChatGPT_client, System_prompt, rule, mr)
    min_model = min(scores, key=scores.get)
    new_row = {
        "Rule": rule,
        "Lowest_Model": min_model,
        "Lowest_Score": scores[min_model],
        "Best_MR": mrs[min_model],
        "Score_GPT": scores[models[0]],
    "Score_Claude": scores[models[1]],
    "Score_Qwen": scores[models[2]],
    }
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
    results.to_csv("lowest_score_models.csv", index=False)
results.to_csv("lowest_score_models.csv", index=False)

#######################################################
#type2
import pandas as pd
import json
from tqdm import tqdm
import re

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
                "person with a disability using a tricycle", "person with a disability using a quadricycle", "child",
                "senior (elderly person)", "person with small children", "bicyclist", "bicycle",
                "bicycle with auxiliary motor", "electric micro-vehicle", "passenger", "driver in front",
                "vehicle behind", "railway employee with flag", "obstacle on the road", "road narrowing",
                "heavy traffic", "rain environment", "snow environment", "mud environment", "ice road", "wet road",
                "fog environment", "heavy smoke environment", "high winds environment", "low lighting environment"]

Expected_Behavior = ["slow down", "turn left", "turn right", "keep current"]

for Num in range(4):
    models = ["GPT", "Claude", "Qwen"]
    data = {}

    # 读取三个模型的 MR 列表
    for model in models:
        df = pd.read_excel(str(Num) + "/" + f"{model}_Traffic_Rules_Results.xlsx")
        data[model] = df["Metamorphic Relation"]
    rules = df["Traffic Rule"]

    results = []


    def parse_json(text):
        return json.loads(text.replace("'", '"'))


    # Prompt 构造（见上方）
    def build_prompt(rule, mrs):
        mr_lines = ""
        for i, mr in enumerate(mrs):
            mr_lines += f"{i + 1}. {mr}\n"
        return f"""
        You are given:
1. The domain-specific language (DSL) used for expressing Metamorphic Relations (MRs), defined as follows:
    - Road Type: {Road_type}
    - Manipulation: {Manipulation}
    - Ego-Vehicle Expected Behavior: {Expected_Behavior}

2. Each MR must strictly follow this format:
    Given the ego-vehicle approaches to <Road Type>, 
    When AUTOMT <adds/replaces> <Manipulation>, 
    Then ego-vehicle should <Expected Behavior>.

**Important**: 
- All elements in the MR (**Road Type**, **Manipulation**, and **Expected Behavior**) must come directly from the above DSL lists. 
- Do not infer new terms or use synonyms that are not listed. 
- If the traffic rule is general and does not specify a particular road type, then 'any roads' from the DSL is acceptable.

You are now given the following traffic rule:

{rule}

And these 3 candidate Metamorphic Relations (MRs):

{mr_lines}

Please select the MR that is most logically consistent with the rule, and adheres strictly to the DSL terms.

Respond only with a JSON object in the following format:
{{"best_option": 1}}
    """.strip()


    # 主循环

    for i in tqdm(range(len(rules))):
        rule = rules[i]
        mrs = [data[model].iloc[i] for model in models]

        prompt = build_prompt(rule, mrs)
        response = chatbot.generate_response(prompt)
        parsed = parse_json(response)

        if parsed and "best_option" in parsed:
            idx = parsed["best_option"] - 1
            best_mr = mrs[idx]
            best_model = models[idx]
            best_mr = best_mr.rstrip(".")
            if "ego-vehicle should stop" in best_mr:
                best_mr = best_mr.replace("ego-vehicle should stop.", "ego-vehicle should slow down.")
                best_mr = best_mr.replace("ego-vehicle should stop", "ego-vehicle should slow down")
            results.append({
                "Rule": rule,
                "Best_MR": best_mr,
                "Model": best_model
            })

    # 保存结果
    pd.DataFrame(results).to_csv(str(Num) + "/" + "lowest_score_models.csv", index=False)