import random
import re
from system import separate_image_paths,save_to_csv,append_to_csv
import os
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
load_dotenv()
os.environ["OPENAI_API_KEY"] = # python 3.9
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import re
import base64
import random



Baseline_2_California = [
  "Given the ego-vehicle is approaching a green light, When a pedestrian enters the crosswalk unexpectedly, Then the ego-vehicle should slow Down.",
  "Given the ego-vehicle is approaching a school zone, When AUTOMT adds children near the roadside, Then the ego-vehicle should slow Down.",
  "Given the ego-vehicle is about to pass a bicyclist, When AUTOMT narrows the lane, Then the ego-vehicle should slow Down.",
  "Given the ego-vehicle is cruising at constant speed, When AUTOMT introduces fog or low-visibility conditions, Then the ego-vehicle should slow Down.",
  "Given the ego-vehicle is navigating a curve, When AUTOMT adds a wet road surface, Then the ego-vehicle should slow Down.",
  "Given the ego-vehicle is following a motorcycle, When AUTOMT adds uneven road conditions ahead, Then the ego-vehicle should slow Down.",
  "Given the ego-vehicle is driving on a mountain road, When AUTOMT introduces an oncoming vehicle on a narrow incline, Then the ego-vehicle should turn right."
]
Baseline_2_German = [
  "Given the ego-vehicle is driving in a 50 km/h zone, When AUTOMT replaces the speed limit sign with a 30 km/h sign, Then the ego-vehicle should Slow Down.",
  "Given the ego-vehicle is traveling under clear weather conditions, When AUTOMT changes the condition to dense fog, Then the ego-vehicle should Slow Down.",
  "Given the ego-vehicle is driving on a road outside school hours, When AUTOMT sets the current time to active school zone hours, Then the ego-vehicle should Slow Down.",
  "Given the ego-vehicle is driving on a highway with no obstructions, When AUTOMT inserts a construction zone with appropriate signage, Then the ego-vehicle should Slow Down.",
  "Given the ego-vehicle is following another vehicle on a dry road When, AUTOMT changes the road surface to icy, Then the ego-vehicle should Slow Down."
]




class California_Components:
    # Road Types
    ROAD_TYPES = [
        "All roads",
        "Intersections",
        "Crosswalks",
        "School Zones",
        "Unmarked Crosswalk",
        "Work Zone"
    ]

    # Adds
    ADDS = [
        # Traffic infrastructure
        "Red Light", "Yellow Light", "Green Light",
        "Red Arrow", "Green arrow",
        "STOP Sign", "YIELD Sign", "5-Sided Sign",
        "Warning Signs", "Crosswalk Markings", "Limit lines",
        # Object
        "Vehicle", "Emergency vehicle", "Tow truck", "Road work vehicle",
        "Pedestrian", "Person using roller skates", "Person using a skateboard",
        "Person with a disability using a wheelchair", "Person with a disability using a tricycle",
        "Person with a disability using a quadricycle", "Child", "Senior (elderly person)",
        "Person with small children", "Bicyclist", "Heavy Traffic"
    ]

    # Replaces
    REPLACES = [
        "Rain", "Snow", "Mud", "Ice", "Wet road",
        "Fog", "Heavy smoke", "High winds", "Low lighting (implied)"
    ]

    # Expected Behavior
    EXPECTED_BEHAVIORS = [
        "Slow down", "Turn left", "Turn right", "Keep current"
    ]

class German_Components:
    # Road Types
    ROAD_TYPES = [
        "All roads",
        "Intersection",
        "Port Area",
        "Field path",
        "Forest path",
        "Footpath",
        "Bicycle path",
        "Industrial area"
    ]

    # Adds - Traffic infrastructure + Object
    ADDS = [
        # Traffic infrastructure
        "St. Andrew's Cross sign",
        "Bus Stop Sign",
        "Start of 30 km/h Zone Sign",
        "Maximum Speed Limit Sign",
        "No Entry Sign",
        "Prohibition for Motor Vehicles Sign",
        "Prohibition for Vehicles of All Kinds Sign",
        "Warning Signs",
        "Right-of-Way Regulation Signs",
        # Objects
        "Stopped vehicle",
        "Oncoming vehicle",
        "Turning vehicle",
        "Rail vehicle",
        "Public transport bus",
        "School bus",
        "Line bus",
        "Motor vehicle",
        "Multi-track motor vehicle",
        "Pedestrian",
        "Bicycle",
        "Bicycle with auxiliary motor",
        "Electric micro-vehicle",
        "Passenger",
        "Driver in front",
        "Vehicle behind",
        "Railway employee with flag",
        "Obstacle on the road",
        "Road narrowing"
    ]

    # Replaces - Environment
    REPLACES = [
        "Fog",
        "Snowfall",
        "Rain"
    ]

    # Ego-Vehicle Expected Behavior
    EXPECTED_BEHAVIORS = [
        "Slow down",
        "Turn left",
        "Turn right",
        "Keep current"
    ]


def generate_random_mr(components_class):
    # 随机选择 Road Type 和 Expected Behavior
    road_type = random.choice(components_class.ROAD_TYPES)
    expected_behavior = random.choice(components_class.EXPECTED_BEHAVIORS)

    # 随机选择 Adds 或 Replaces
    if random.choice(["adds", "replaces"]) == "adds":
        manipulation = random.choice(components_class.ADDS)
        mr = (
            f"Given the ego-vehicle approaches to {road_type.lower()}, "
            f"When AUTOMT adds {manipulation.lower()}, "
            f"Then ego-vehicle should {expected_behavior.lower()}."
        )
    else:
        manipulation = random.choice(components_class.REPLACES)
        mr = (
            f"Given the ego-vehicle approaches to {road_type.lower()}, "
            f"When AUTOMT replaces the environment into {manipulation.lower()}, "
            f"Then ego-vehicle should {expected_behavior.lower()}."
        )

    return mr

def extract_unique_road_types(rules,type=0):
    road_types = set()
    if type ==0:
        pattern = r"Given the ego-vehicle approaches to ([^,]+),"
    else:
        pattern = r"Given the ego-vehicle is ([^,]+),"

    for rule in rules:
        match = re.search(pattern, rule)
        if match:
            road_types.add(match.group(1).strip())

    return sorted(road_types)




def Baseline1(Excel_name):

    def process_path(paths, mr_LLM, type=1,infos=None):
        video_length = 10
        results = []
        total_paths = len(paths)
        for i in range(0, total_paths, video_length):
            time.sleep(5)
            video_frames = []
            batch_paths = paths[i:min(i + video_length, total_paths)]
            for path in batch_paths:
                local_path = os.path.join("Data","test", os.path.basename(path))
                video_frames.append(local_path)
           # Matched_MR = mr_LLM.analyze_scene_to_mr(video_frames[0])
            try:
                Matched_MR = mr_LLM.analyze_scene_to_mr(video_frames[0],infos[i + 5])
            except Exception as e:
                print(f"Error analyzing scene: {video_frames[0]}")
                Matched_MR = None
            print(Matched_MR)
            results.append({
                "frames": video_frames[0],
                "analysis": 0,
                "matched_result": Matched_MR,
                "Wrong_number": 0
            })
        return results

    udacity_paths, udacity_infos, a2d2_paths, a2d2_infos = separate_image_paths()
    mr_LLM = ImageAnalyzer_GPT(client=OpenAI())

    california_results = process_path(udacity_paths, mr_LLM, "California",udacity_infos)
    save_to_csv(california_results, Excel_name+".csv")
    print(1)
    german_results = process_path(a2d2_paths,mr_LLM, "German",a2d2_infos)
    append_to_csv(german_results, Excel_name+".csv")


def Baseline2(Excel_name):
    def process_path(paths, mr_LLM, type=1, infos=None):
        video_length = 10
        results = []
        total_paths = len(paths)
        for i in range(0, total_paths, video_length):
            if i<=167:
                traffic_rule = """A green traffic signal light means GO.
"STOP Sign
stop sign
Make a full stop before entering the crosswalk or at the limit line. If there is no limit line or crosswalk, stop before entering the intersection. Check traffic in all directions before proceeding."
"Red YIELD Sign
red yield sign
Slow down and be ready to stop to let any vehicle, bicyclist, or pedestrian pass before you proceed."
"5-sided Sign

You are near a school. Drive slowly and stop for children in the crosswalk."
The vehicle that arrives to the intersection first has the right-of-way. However, if a vehicle gets to the intersection at the same time as you, give the right-of-way to the vehicl.
The vehicle that arrives to the intersection first has the right-of-way. However, if a v pedestrian gets to the intersection at the same time as you, give the right-of-way  pedestrian on your right.
The vehicle that arrives to the intersection first has the right-of-way. However, if a bicyclist gets to the intersection at the same time as you, give the right-of-way to the bicyclist on your right.
When there is a pedestrian crossing a roadway with or without a crosswalk, you must use caution, reduce your speed, or stop to allow the pedestrian to safely finish crossing.
When there is A person traveling on something other than a vehicle or bicycle. This includes roller skates, a skateboard, etc. crossing a roadway with or without a crosswalk, you must use caution, reduce your speed, or stop to allow the pedestrian to safely finish crossing.
When there is A person with a disability using a tricycle, quadricycle, or wheelchair for transportation. crossing a roadway with or without a crosswalk, you must use caution, reduce your speed, or stop to allow the pedestrian to safely finish crossing.
Do not pass a vehicle stopped at a crosswalk. You may not be able to see a pedestrian crossing the street.
Always allow pedestrians enough time to safely cross a street as some groups such as seniors, people with small children may require extra time.
Always allow pedestrians enough time to safely cross a street as some groups such as seniors, people with disabilities may require extra time.
Pedestrians have the right-of-way in marked or unmarked crosswalks. If there is a limit line before the crosswalk, stop at the limit line and allow pedestrians to cross the street.
"You will see warning signs and message boards when there are workers, slow-moving equipment, and closed lanes ahead.

Go through the work zone carefully by:

Slowing down."
"Drivers must move over and slow down for emergency and road work vehicles. These include:
Stopped road work vehicles displaying emergency flashing or amber warning lights."
"Drivers must move over and slow down for emergency and road work vehicles. These include:
Stationary emergency vehicles or tow trucks displaying flashing amber warning lights."
You must drive slower when there is heavy traffic 
You must drive slower when there is bad weather
"Slippery Roads
Rain, snow, or mud can make the roads slippery. Drive more slowly than you would on a dry road. Adjust your speed for different conditions:

Wet road: Reduce your speed by 5 to 10 mph."
"Slippery Roads
Rain, snow, or mud can make the roads slippery. Drive more slowly than you would on a dry road. Adjust your speed for different conditions:
Packed snow: Reduce your speed by half."
"Slippery Roads
Rain, snow, or mud can make the roads slippery. Drive more slowly than you would on a dry road. Adjust your speed for different conditions:
Ice: Reduce your speed to no more than 5 mph."
"Hydroplaning
Hydroplaning occurs when driving in wet or rainy conditions. Hydroplaning is a vehicle riding on water because the tires lose all contact with the road. A slight change of direction, applying the brakes, or a gust of wind could throw the vehicle into a skid. To avoid hydroplaning:

Drive slowly."
"High Winds
High winds can be a hazard while driving. This is especially true for larger vehicles such as trucks, campers, and vehicles with trailers. When driving in high winds:

Reduce your speed. This gives you better control over your vehicle. You will have more time to react if your vehicle gets hit by a strong gust of wind."
"Fog or Heavy Smoke
It is best to avoid driving in heavy fog or smoke. Consider postponing your trip until the fog clears. If you must drive in heavy fog or smoke:

Drive slowly."
"Law Enforcement Traffic Breaks
During a traffic break, the officer turns on their emergency lights and slowly weaves across lanes. You should:

Turn on your emergency flashers to warn other drivers.
Slowly decrease your speed to the same speed as the officer. Do not brake suddenly unless necessary to avoid a collision. Keep a safe distance from the patrol vehicle ahead of you."
Do not turn at a red arrow. Remain stopped until a green traffic signal light or green arrow appears.
RIGHT LANE MUST TURN RIGHT sign: Vehicles driving in the right lane must turn right at the next intersection unless the sign indicates a different turning point.
Red and White Regulatory Sign DO NOT ENTER Follow the sign’s instruction. For example, DO NOT ENTER means do not enter the road or ramp where the sign is posted.
WRONG WAY Sign WRONG WAY If you enter a roadway against traffic, DO NOT ENTER and WRONG WAY signs may be posted. When it is safe, back out or turn around. If you are driving at night, you will know you are going the wrong way if the road reflectors shine red in your headlights.
Red Circle with a Red Line Through It The picture inside the circle shows what you cannot do and may be shown with words
Yellow and Black Circular Sign or X-shaped Sign R R You are approaching a railroad crossing. Look, listen, slow down, and prepare to stop. Let any trains pass before you proceed.
5-sided Sign You are near a school. Drive slowly and stop for children in the crosswalk.
Diamond-shaped Sign Warns you of specific road conditions and dangers ahead.
Warning Signs Warns of conditions related to pedestrians, bicyclists, schools, playgrounds, school buses, and school passenger loading zones. For more information about signs, visit dot.ca.gov/programs/safety
Before entering an intersection, look left, right, and ahead to check for vehicles, bicyclists, and pedestrians. Be prepared to slow down and stop if necessary. Pedestrians always have the right-of-way. Here are some right-of-way rules at intersections
Without STOP or YIELD signs: The vehicle that arrives to the intersection first has the right-of-way. However, if a vehicle, pedestrian, or bicyclist gets to the intersection at the same time as you, give the right-of-way to the vehicle, pedestrian, or bicyclist on your right. If you approach a stop sign and there is a stop sign on all four corners, stop first and proceed as above
Entering traffic: When entering traffic, you must proceed with caution and yield to the traffic already occupying the lanes. It is against the law to stop or block an intersection where there is not enough space to completely cross before the traffic signal light turns red.
Yield to all traffic already in the roundabout
These are considered pedestrians or vulnerable road users:  A person walking.
"These are considered pedestrians or vulnerable road users:  A person traveling on something other than a vehicle or bicycle. This
includes roller skates, a skateboard, etc."
These are considered pedestrians or vulnerable road users:  A person with a disability using a tricycle, quadricycle or wheelchair for transportation.
"When there is a pedestrian crossing a roadway with or without a
crosswalk, you must use caution, reduce your speed, or stop to allow the
pedestrian to safely finish crossing. "
"Do not pass a vehicle stopped at a crosswalk. You may not be able to
see a pedestrian crossing the street"
"If a pedestrian makes eye contact with you, they are ready to cross
the street. Yield to the pedestrian"
"Always allow pedestrians enough time to safely cross a street as some
groups such as seniors, people with small children, and people with
disabilities may require extra time."
"Some crosswalks have flashing lights. Whether or not the lights are
flashing, look for pedestrians and be prepared to stop. "
"Pedestrians have the right-of-way in marked or unmarked crosswalks. If
there is a limit line before the crosswalk, stop at the limit line and allow
pedestrians to cross the street. "
"Pedestrians using guide dogs or white canes have the right-of-way at all
times. These pedestrians are partially or totally blind. Be careful when you
are turning or backing up. This is particularly important if you are driving
a hybrid or electric vehicle because blind pedestrians rely on sound to
know there is a vehicle nearby. "
"Do not stop in the middle of a crosswalk. This could force a blind
pedestrian to walk into traffic outside of the crosswalk."
Yield to emergency vehicles
When approaching a stationary emergency vehicle with flashing\nemergency signal lights (hazard lights), move over and slow down'
It is against the law to follow within 300 feet of any fire engine, law\nenforcement vehicle, ambulance, or other emergency vehicle when their\nsiren or flashing lights are o
Near Animals\nIf you see a sign with a picture of an animal, watch for animals on or near\nthe road
'If you see animals or livestock near the road, slow down or stop\nand proceed when it is safe
Drivers must move over and slow down for emergency and road work
Some school buses flash yellow lights when preparing to stop to let children off the bus. The yellow flashing lights warn you to slow down and prepare to stop.
When the bus flashes red lights (located at the top, front, and back of the bus), you must stop from either direction until the children are safely across the street and the lights stop flashing. Remain stopped while the red lights are flashing. If you fail to stop, you may be fined up to $1,000 and your driving privilege could be suspended for one year.
Near Railroad or Light Rail Tracks Flashing red warning lights indicate you must stop and wait. Do not proceed over the railroad tracks until the red lights stop flashing, even if the gate rises
Near Railroad or Light Rail Stop, look, and listen. If you see a train coming or hear a horn or bell, do not cross. Many crossings have multiple tracks. Look in both directions and only cross when it is safe
Tailgating makes it harder for you to see the road ahead because the vehicle in front of you blocks your view. You will not have enough time to react if the driver in front of you brakes suddenly. Use the three-second rule to ensure a safe following distance and avoid a collision. Following other vehicles at a safe distance giv
Highway construction can take place at night. Reduce your speed in highway construction zones
When you leave a brightly lit place, drive slowly until your eyes adjust to the darkness
When a vehicle with one light drives toward you, drive as far to the right as possible. It could be a bicyclist, motorcyclist, or vehicle with a missing headlight.
Slow down at the first sign of rain on the road.
Slow down at the first sign of  drizzle on the road.
Slow down at the first sign of snow on the road.
Slow down and let the oncoming vehicle pass.
If you see a vehicle’s emergency flashers ahead, slow down.
If you are in a collision: • You must stop.
"""
            else:
                traffic_rule = """
                If visibility is reduced to less than 50 meters due to fog, the speed must not exceed 50 km/h, unless a lower speed is required.
If visibility is reduced to less than 50 meters due to snowfall, the speed must not exceed 50 km/h, unless a lower speed is required.
If visibility is reduced to less than 50 meters due to rain, the speed must not exceed 50 km/h, unless a lower speed is required.
The distance to a vehicle ahead must generally be large enough so that one can still stop behind it if it suddenly brakes. The driver in front must not brake sharply without a compelling reason.
"
At a road narrowing, an obstacle on the road, or a stopped vehicle, anyone who wants to pass on the left must let oncoming vehicles go through. The first rule does not apply if the right of way is regulated differently by traffic signs (Signs 208, 308). If one needs to pull out, attention must be paid to the traffic behind, and both pulling out and merging back in—like overtaking—must be signaled."
When passing public transport buses that are stopped at bus stops (Sign 224) with their hazard lights on, vehicles must only pass at walking speed and maintain a distance that ensures the safety of passengers is not compromised. This precaution helps prevent accidents involving passengers entering or exiting the buses.
Anyone wishing to turn must let oncoming vehicles pass, including rail vehicles, even if they are traveling on or alongside the roadway in the same direction. This also applies to line buses and other vehicles using designated special lanes. Special consideration must be given to pedestrians; if necessary, one must wait.
Anyone wishing to turn must let oncoming vehicles pass, including bicycles with auxiliary motors, even if they are traveling on or alongside the roadway in the same direction. This also applies to line buses and other vehicles using designated special lanes. Special consideration must be given to pedestrians; if necessary, one must wait.
Anyone wishing to turn must let oncoming vehicles pass, including bicycles, even if they are traveling on or alongside the roadway in the same direction. This also applies to line buses and other vehicles using designated special lanes. Special consideration must be given to pedestrians; if necessary, one must wait.
Anyone wishing to turn must let oncoming vehicles pass, including electric micro-vehicles, even if they are traveling on or alongside the roadway in the same direction. This also applies to line buses and other vehicles using designated special lanes. Special consideration must be given to pedestrians; if necessary, one must wait.
" Anyone wishing to turn left must allow oncoming vehicles that want to turn right to pass. Oncoming vehicles wanting to turn left must turn in front of each other, unless the traffic situation or the design of the intersection requires waiting until the vehicles have passed each other.
Start of a 30 km/h Zone"" Sign: Command or Prohibition

Anyone driving a vehicle must not exceed the maximum speed limit indicated within this zone.

Explanation: Along with this sign, speed limits of less than 30 km/h can be imposed in traffic-calmed business areas. This helps ensure that traffic moves slowly and safely through areas with high pedestrian activity or where vehicles frequently interact with other road users."
"At railroad crossings with a St. Andrew's cross (Sign 201),
At railroad crossings over footpaths, field paths, forest paths, or bicycle paths,
In port and industrial areas if the entrance displays a St. Andrew's cross with the additional sign “Port area, rail vehicles have priority” or “Industrial area, rail vehicles have priority”.
Road traffic must approach such crossings at a moderate speed. Those driving a vehicle must not overtake other motor vehicles from the sign 151, 156 up to and including the intersection area of rail and road."
(3) If the railroad crossing cannot be crossed swiftly and without delay due to road traffic, one must wait before the St. Andrew's cross.
(4) Those using a footpath, field path, forest path, or bicycle path must behave accordingly at railroad crossings without a St. Andrew's cross.
(5) At railroad crossings without the priority of rail vehicles, one must wait at a safe distance if a railway employee with a white-red-white flag or a red light signals to stop. If yellow or red light signals are given, § 37 paragraph 2 number 1 applies accordingly.
At bus stops (Sign 224) where public transport buses, trams, and designated school buses are stopping, vehicles, including those in the oncoming traffic, may only pass cautiously.
When passengers are boarding, vehicles may only pass on the right at walking speed and at such a distance that passenger safety is not jeopardized. Passengers must not be hindered. If necessary, drivers must wait.
When passengers are alighting, vehicles may only pass on the right at walking speed and at such a distance that passenger safety is not jeopardized. Passengers must not be hindered. If necessary, drivers must wait.
Public transport buses and designated school buses approaching a stop (Sign 224) with hazard lights activated must not be overtaken.
At bus stops (Sign 224) where public transport buses and designated school buses are stopping with hazard lights on, vehicles may only pass at walking speed and at such a distance that the safety of passengers is not compromised. This walking speed also applies to oncoming traffic on the same road. Passengers must not be hindered. If necessary, drivers must wait.
Other vehicles must allow public transport buses and school buses to depart from designated stops. If necessary, other vehicles must wait.
"
""Maximum Speed Limit"" Sign: Command or Prohibition

A person driving a vehicle must not exceed the speed limit indicated on the sign."
Yellow indicates: "Wait before the intersection for the next signal."
"
""No Entry"" Sign: A person driving a vehicle is not permitted to enter the roadway for which the sign is designated.

Explanation: The sign is positioned on the right side of the roadway to which it applies, or on both sides of that roadway."
Red orders: "Stop before the intersection."
"
Prohibition for Motor Vehicles sign: Command or Prohibition Prohibition for motor vehicles and other multi-track motor vehicles."
Warning signs call for increased attention, especially for reducing speed in anticipation of a hazardous situation.
Prohibition for Vehicles of All Kinds sign: Command or Prohibition Prohibition for vehicles of all kinds. This sign does not apply to hand carts, and contrary to § 28 paragraph 2, it also does not apply to riders, leaders of horses, and drivers and leaders of livestock. Motorcycles and bicycles may be pushed.
Anyone driving a vehicle must behave in such a way towards children— particularly by reducing speed and being prepared to brake — that any endangerment to these road users is ruled out.
Anyone driving a vehicle must behave in such a people in need of assistancs — particularly by reducing speed and being prepared to brake — that any endangerment to these road users is ruled out.
Anyone driving a vehicle must behave in such a elderly individuals — particularly by reducing speed and being prepared to brake — that any endangerment to these road users is ruled out.
Yellow: "Wait for the next signal before the intersection."
Red: "Stop before the intersection."
A black arrow on red requires stopping; a
a black arrow on yellow requires waiting,
"Prescribed Minimum Speed Anyone driving a vehicle must not drive slower than the indicated minimum speed, unless road, traffic, visibility, or weather conditions require it.
It is prohibited to use a lane marked in this way with vehicles that are not able or allowed to travel at the specified minimum speed."
Anyone who would normally be allowed to proceed under traffic rules or has the right of way must yield if the traffic situation requires it.

                """
            time.sleep(5)
            video_frames = []
            batch_paths = paths[i:min(i + video_length, total_paths)]
            for path in batch_paths:
                local_path = os.path.join("Data", "test", os.path.basename(path))
                video_frames.append(local_path)
            # Matched_MR = mr_LLM.analyze_scene_to_mr(video_frames[0])
            try:
                Matched_MR = mr_LLM.analyze_scene_to_mr_rule(video_frames[0], infos[i + 5],traffic_rule)
            except Exception as e:
                print(f"Error analyzing scene: {video_frames[0]}")
                Matched_MR = None
            print(Matched_MR)
            results.append({
                "frames": video_frames[0],
                "analysis": 0,
                "matched_result": Matched_MR,
                "Wrong_number": 0
            })
        return results

    udacity_paths, udacity_infos, a2d2_paths, a2d2_infos = separate_image_paths()
    mr_LLM = ImageAnalyzer_GPT(client=OpenAI())

    california_results = process_path(udacity_paths, mr_LLM, "California", udacity_infos)
    save_to_csv(california_results, Excel_name + ".csv")
    print(1)
    german_results = process_path(a2d2_paths, mr_LLM, "German", a2d2_infos)
    append_to_csv(german_results, Excel_name + ".csv")


mr_list = [
    "Given the ego-vehicle approaches to any roads, When AUTOMT adds a pedestrian on the roadside, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT adds a speed limit sign on the roadside, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT replaces time into night, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT replaces weather into snow, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT replaces sunny with rainy, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT adds a vehicle in front of the ego-vehicle, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT adds a cyclist in front of the ego-vehicle, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT adds a red light on the roadside, Then ego-vehicle should slow down.",
    "Given the ego-vehicle approaches to any roads, When AUTOMT adds a green light on the roadside, Then ego-vehicle should keep current."
]

def Baseline3(Excel_name):
    def process_path(paths, infos, RAG_location, type=1):
        video_length = 10
        results = []
        total_paths = len(paths)
        for i in range(0, total_paths, video_length):
            video_frames = []
            # 获取当前批次的图片路径（最多10张）
            batch_paths = paths[i:min(i + video_length, total_paths)]
            for path in batch_paths:
                local_path = os.path.join("C:/Users/Administrator/Desktop/ITMT_EXP/Data/test", os.path.basename(path))
                video_frames.append(local_path)
            results.append({
                "frames": video_frames[0],
                "analysis": 0,
                "matched_result": random.choice(mr_list),
                "Wrong_number": 0
            })
        return results

    udacity_paths, udacity_infos, a2d2_paths, a2d2_infos = separate_image_paths()

    california_results = process_path(udacity_paths, udacity_infos, "California")
    save_to_csv(california_results, Excel_name+".csv")
    print(1)
    german_results = process_path(a2d2_paths,a2d2_infos, "German")
    append_to_csv(german_results, Excel_name+".csv")

def Baseline(Excel_name,baseline_num):

    def process_path(paths, mr_LLM, RAG_location, type=1):
        Components = Baseline_1
        if RAG_location == "California":
            if baseline_num==2:
                Components = Baseline_2_California
        else:
            if baseline_num == 2:
                Components = Baseline_2_German
        mr_LLM.update_MRs(Components)
        video_length = 10
        results = []
        total_paths = len(paths)
        for i in range(0, total_paths, video_length):
            video_frames = []
            # 获取当前批次的图片路径（最多10张）
            batch_paths = paths[i:min(i + video_length, total_paths)]
            for path in batch_paths:
                local_path = os.path.join("C:/Users/Administrator/Desktop/ITMT_EXP/Data/test", os.path.basename(path))
                video_frames.append(local_path)
            Matched_MR = mr_LLM.analyze_media(video_frames)
            results.append({
                "frames": video_frames[0],
                "analysis": 0,
                "matched_result": Matched_MR,
                "Wrong_number": 0
            })
            print(Matched_MR)
        return results

    udacity_paths, udacity_infos, a2d2_paths, a2d2_infos = separate_image_paths()
    mr_LLM = Matching_MR()

    california_results = process_path(udacity_paths, mr_LLM, "California")
    save_to_csv(california_results, Excel_name+".csv")
    print(1)
    german_results = process_path(a2d2_paths,mr_LLM, "German")
    append_to_csv(german_results, Excel_name+".csv")




class ImageAnalyzer_GPT:
    def __init__(self, client, model="gpt-4.1-mini-2025-04-14"):
        self.client = client
        self.model = model

    def _load_image_base64(self, image_path):
        """读取本地 PNG 图像并转为 base64 数据 URI。"""
        image_data = Path(image_path).read_bytes()
        base64_str = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"

    def analyze_scene_to_mr(self, image_path,vheicle_info=None):
        image_data_uri = self._load_image_base64(image_path)

        prompt = (
            "You are an expert in traffic rules and scene understanding.\n"
            "Metamorphic Testing (MT) is a method used in autonomous vehicle testing.\n"
            "It defines how the behavior of a system should change when its inputs are changed in a specific way.\n\n"
            "Given an image of a traffic scene, generate a Metamorphic Relation (MR) in the format:\n"
            "Given the ego-vehicle approaches to [road type]\n"
            "When AUTOMT [modification]\n"
            "Then ego-vehicle should [expected behavior]\n\n"
            "The [expected behavior] must be one of the following four actions:\n"
            "- slow down\n"
            "- keep current\n"
            "- turn left\n"
            "- turn right\n\n"
            "Additional context about the ego-vehicle:\n"
            f"{vheicle_info}\n\n"
            "Example:\n"
            "Given the ego-vehicle approaches to an intersection\n"
            "When AUTOMT adds a steady red light on the roadside\n"
            "Then ego-vehicle should slow down\n\n"
            "Now generate an MR for the image below. Only output the MR without explanation."
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_uri}
                    ]
                }
            ],
            reasoning={},
            tools=[],
            temperature=0.1,
            max_output_tokens=512,
            top_p=1,
            store=False
        )

        return response.output[0].content[0].text

    def analyze_scene_to_mr_rule(self, image_path,traffic_rule,vheicle_info=None):
        image_data_uri = self._load_image_base64(image_path)

        prompt = (
            "You are an expert in traffic rules and scene understanding.\n"
            "Metamorphic Testing (MT) is a method used in autonomous vehicle testing.\n"
            "Traffic rules: "
            "It defines how the behavior of a system should change when its inputs are changed in a specific way.\n\n"
            "Given an image of a traffic scene and traffic rules, extract a Metamorphic Relation (MR) from traffic rule in the format:\n"
            "Given the ego-vehicle approaches to [road type]\n"
            "When AUTOMT [modification]\n"
            "Then ego-vehicle should [expected behavior]\n\n"
            "The [expected behavior] must be one of the following four actions:\n"
            "- slow down\n"
            "- keep current\n"
            "- turn left\n"
            "- turn right\n\n"
            "Additional context about the ego-vehicle:\n"
            f"{vheicle_info}\n\n"
            "Example:\n"
            "Given the ego-vehicle approaches to an intersection\n"
            "When AUTOMT adds a steady red light on the roadside\n"
            "Then ego-vehicle should slow down\n\n"
            "Now extract an MR from traffic rules for the image below. Only output the MR without explanation."

        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_uri}
                    ]
                }
            ],
            reasoning={},
            tools=[],
            temperature=0.1,
            max_output_tokens=512,
            top_p=1,
            store=False
        )

        return response.output[0].content[0].text

    def match_scene_to_mr(self, image_path,MRs,vheicle_info=None):
        image_data_uri = self._load_image_base64(image_path)
        print({vheicle_info})
        prompt = (
            "You are an expert in traffic scene understanding and Metamorphic Testing (MT).\n"
            "Metamorphic Testing is used in autonomous vehicle systems to verify that\n"
            "the system behaves correctly when a specific change is made to the input.\n"
            "Each Metamorphic Relation (MR) describes:\n"
            "- the original situation,\n"
            "- a specific modification (manipulation), and\n"
            "- the expected behavior of the ego-vehicle.\n\n"
            "Example MR:\n"
            "Given the ego-vehicle approaches to an intersection\n"
            "When AUTOMT adds a steady red light on the roadside\n"
            "Then ego-vehicle should slow down\n\n"
            "Your task:\n"
            "Analyze the image below, and from the list of candidate Metamorphic Relations (MRs),\n"
            "choose the **one and only one** MR that best matches the traffic scene in the image.\n"
            "**You must choose one full MR from the list exactly as written. Do not invent or modify any MR.**\n\n"
            "Candidate MRs:\n"
            f"{MRs}\n\n"
            "Additional context about the ego-vehicle:\n"
            f"{vheicle_info}\n\n"
            "Now analyze the image and respond with the most appropriate MR from the list above. Only output the selected MR, nothing else."
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_data_uri}
                    ]
                }
            ],
            reasoning={},
            tools=[],
            temperature=0.1,
            max_output_tokens=512,
            top_p=1,
            store=False
        )

        return response.output[0].content[0].text


class Matching_MR:
    def __init__(self, VLM="Qwen"):
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
        self.vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.selected_mrs = []  # Store previously selected MRs
        self.max_history = 3  # Maximum number of MRs to keep in history
        self.Wrong = 0
    def update_MRs(self,MRs):
        self.MRs = MRs
        self.road_types = extract_unique_road_types(MRs,type=1)

    def analyze_media(self, images):
        if self.VLM == "Qwen":
            prompt = (
                f"Analyze this driving scene. Describe the road types (must be one of the following: {', '.join(self.road_types)}), "
                f"'all roads' can be used if no specific type fits). "
                f"Reply format: road network: <road_type>"
            )

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
            raw_result = output_text[0].strip()
            # Step 1: Remove prefix if present
            if raw_result.lower().startswith("road network:"):
                road_type = raw_result[len("road network:"):].strip()
            else:
                road_type = raw_result

            # Step 2: Validate road_type
            if road_type not in self.road_types:
                road_type = random.choice(self.road_types)

            matching_MRs = [mr for mr in self.MRs if road_type in mr]

            # Step 4: Randomly select one if there are matches
            if matching_MRs:
                selected_MR = random.choice(matching_MRs)
            else:
                # fallback: randomly select any MR
                selected_MR = random.choice(self.MRs)

            return selected_MR

class ImageAnalyzer_Qwen:
    def __init__(self, device="cuda:0"):
        self.VLM = "Qwen"
        self.device = device
        self.vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
            device_map={"": self.device}
        )
        self.vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    def analyze_scene_to_mr(self, image_path):
        prompt = (
            "You are an expert in traffic rules and scene understanding.\n"
            "Metamorphic Testing (MT) is a method used in autonomous vehicle testing.\n"
            "It defines how the behavior of a system should change when its inputs are changed in a specific way.\n\n"
            "Given an image of a traffic scene, generate a Metamorphic Relation (MR) in the format:\n"
            "Given the ego-vehicle approaches to [situation]\n"
            "When AUTOMT [modification]\n"
            "Then ego-vehicle should [expected behavior]\n\n"
            "The [expected behavior] must be one of the following four actions:\n"
            "- slow down\n"
            "- keep current\n"
            "- turn left\n"
            "- turn right\n\n"
            "Example:\n"
            "Given the ego-vehicle approaches to an intersection\n"
            "When AUTOMT adds a steady red light on the roadside\n"
            "Then ego-vehicle should slow down\n\n"
            "Now generate an MR for the image below. Only output the MR without explanation."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},  # 图像路径字符串
                    {"type": "text", "text": prompt}
                ]
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
        raw_result = output_text[0].strip()

        return raw_result

"""prompt = (
    "This is a driving scene. "
    "Determine whether the visual content matches the following driving scenario description. "
    "Be tolerant to minor mismatches, but focus on key elements like road type, objects, and traffic flow. "
    "Reply only with 'Yes' if it is a reasonable match, or 'No' if it is clearly unrelated.\n\n"
    f"Scenario description: {scene_description}"
)"""



#print(generate_random_mr(California_Components))

#analyzer = ImageAnalyzer_Qwen()
for i in range(4):
    folder_path = str(i)
    os.makedirs(folder_path, exist_ok=True)
    if i ==0:
        Baseline2(Excel_name=str(i) + "/" + "Baseline2")
    else:
        Baseline3(Excel_name=str(i)+"/"+"Baseline3")
        Baseline1(Excel_name=str(i)+"/"+"Baseline1")
        Baseline2(Excel_name=str(i)+"/"+"Baseline2")
