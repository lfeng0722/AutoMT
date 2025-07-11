import pandas as pd
import re

# 通用正则：支持句尾有标点或无标点，句中可以逗号或中文逗号分隔
pattern = re.compile(
    r'Given\s+(.*?)[,，]?\s+When\s+(.*?)[,，]?\s+Then\s+(.*?)\s*$',
    re.IGNORECASE
)

for Num in range(4):
    print(f"\n====== 处理模型 {Num} 的 MR 文件 ======")
    df = pd.read_csv(f"{Num}/lowest_score_models.csv", encoding="latin1")
    Best_MR = df["Best_MR"]

    # 划分 German 和 California
    German = Best_MR[0:38]
    California = Best_MR[38:110]

    # === California 处理 ===
    california_df = pd.DataFrame()
    california_df["index"] = range(len(California))
    california_df["MR"] = California.tolist()

    california_df['road type'] = ""
    california_df['modification action'] = ""
    california_df['expected behavior'] = ""

    for i, text in enumerate(california_df["MR"]):
        text = str(text).strip().replace("\n", " ").replace("\r", "")
        match = pattern.search(text)
        if match:
            california_df.at[i, 'road type'] = match.group(1).strip()
            california_df.at[i, 'modification action'] = match.group(2).strip()
            california_df.at[i, 'expected behavior'] = match.group(3).strip()
        else:
            print(f"[未匹配 California] 行 {i}: {text}")

    california_df.drop(columns=["MR"], inplace=True)
    california_df.to_csv(f"{Num}/RAG_California.csv", index=False)

    pd.DataFrame({
        "index": range(len(California)),
        "MR": California
    }).to_csv(f"{Num}/RAG_California_1.csv", index=False)

    # === German 处理 ===
    german_df = pd.DataFrame()
    german_df["index"] = range(len(German))
    german_df["MR"] = German.tolist()

    german_df['road type'] = ""
    german_df['modification action'] = ""
    german_df['expected behavior'] = ""

    for i, text in enumerate(german_df["MR"]):
        text = str(text).strip().replace("\n", " ").replace("\r", "")
        match = pattern.search(text)
        if match:
            german_df.at[i, 'road type'] = match.group(1).strip()
            german_df.at[i, 'modification action'] = match.group(2).strip()
            german_df.at[i, 'expected behavior'] = match.group(3).strip()
        else:
            print(f"[未匹配 German] 行 {i}: {text}")

    german_df.drop(columns=["MR"], inplace=True)
    german_df.to_csv(f"{Num}/RAG_German.csv", index=False)

    pd.DataFrame({
        "index": range(len(German)),
        "MR": German
    }).to_csv(f"{Num}/RAG_German_1.csv", index=False)
import pandas as pd
import re

# ------------ California 数据处理 ------------

# 假设 California 是 list of strings，格式统一为自然语言句式
california_df = pd.DataFrame()
california_df['index'] = range(len(California))
california_df['MR'] = [item[0] if isinstance(item, (list, tuple)) else item for item in California]

pattern = r"Given the ego-vehicle approaches to (.*?), when AUTOMT (.*?), then ego-vehicle should (.*)"

for i, text in enumerate(california_df['MR']):
    match = re.search(pattern, text.strip(), re.IGNORECASE)
    if match:
        california_df.at[i, 'Road Network'] = match.group(1).strip().lower()
        california_df.at[i, 'modification action'] = match.group(2).strip().lower()
        california_df.at[i, 'vehicle maneuver'] = match.group(3).strip().lower()

california_df.to_csv('RAG_California.csv', index=False)

# ------------ German 数据处理 ------------

# 假设 German 是 list of strings，格式也为自然语言句式
german_df = pd.DataFrame()
german_df['index'] = range(len(German))
german_df['MR'] = German

pattern = r"Given the ego-vehicle approaches to (.*?), when AUTOMT (.*?), then ego-vehicle should (.*)"

for i, text in enumerate(german_df['MR']):
    match = re.search(pattern, text.strip(), re.IGNORECASE)
    if match:
        german_df.at[i, 'Road Network'] = match.group(1).strip().lower()
        german_df.at[i, 'modification action'] = match.group(2).strip().lower()
        german_df.at[i, 'vehicle maneuver'] = match.group(3).strip().lower()

german_df.to_csv('RAG_German.csv', index=False)
