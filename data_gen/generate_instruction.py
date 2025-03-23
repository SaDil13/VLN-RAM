import os
import json
import requests
import base64
import os
import random
import shutil
# random.seed(0)
from tqdm import tqdm
import argparse
import time
from transformers import AutoTokenizer  
import copy

api_url = '' # specify your openai api url
api_key = '' # specify your openai api key
headers={
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

sys_temperature = 0.8
sys_presence_penalty = 0

def post_request(api_url, data, headers):
    while True:
        try:
            response = requests.post(api_url, data=json.dumps(data), headers=headers, timeout=10)
            return response
        except requests.exceptions.Timeout:
            print("last request timeout, retrying...")
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(str(e))
            print("failed to connect, retrying...")
            time.sleep(10)


def get_query_gpt35(prompt):
    data = {
        "model":'gpt-3.5-turbo', 
        "messages":[
                {"role":"user", "content":prompt}
        ],
        "temperature": sys_temperature
        ,"presence_penalty": sys_presence_penalty
    }   
        
    response = post_request(api_url, data=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        # print(result['choices'][0]['message']['content'])
        return result['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code} - {response.text}")



def r2r_instr_gen_step1():
    r2r_train_path = '' # the original train json file
    r2r_description_path = '' # description json file
    output_path = '' #specify your output path
    with open(r2r_train_path, 'r') as fre:
        r2r_train_data = json.load(fre)
    with open(r2r_description_path, 'r') as fr1:
        r2r_description_data = json.load(fr1)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    sampled_files = r2r_train_data
    for i in tqdm(range(len(sampled_files))):
        temp_dic = {}
        path_id = sampled_files[i]['path_id']
        instructions = sampled_files[i]['instructions']
        landmark_list = sampled_files[i]['landmark_list'] # you can get the landmark_list by chatgpt
        output_file_path = os.path.join(output_path, 'pathid_'+str(path_id)+'.json')
        if os.path.exists(output_file_path):
            print('skip')
            continue
        temp_dic['path_id'] = path_id
        temp_instr = []
        new_gt = {}
        for cn in range(1,4,1):
            gt_key = str(path_id) + '_rwt_' + str(cn)
            gt_actions = r2r_description_data[gt_key]
            new_gt[gt_key] = gt_actions
            sequential_scene_description = ''
            objs = ''
            for k, v in gt_actions.items():
                step_idx = k.split('$')[0]
                sequential_scene_description += 'Step '+step_idx+':'+v+'.\t'  
            for k, v in landmark_list[str(path_id)+'_'+str(cn-1)].items():
                objs += 'Step '+k+':'+v+'.\t' 
            prompt_4 = f"""Rewrite the original instruction according to the observation descriptions given. The rewritten instruction should satisfy the following principles: 1) The rewritten instruction should replace the object (scene) described in the given instruction with a specific object (scene) mentioned in the new trajectory information according to the matched step information. The rewritten instruction should not contain the objects mentioned in the given instruction. 2) The rewritten instruction must have similar length with the given instruction. 3) The rewritten instruction must not contain the word "Step". 4) The rewritten instruction should change synonyms of verbs in the original instruction, like changing "turn left" to "take a left to". \nHere is an example.\nLandmarks in original trajectory: Step1: workshop; Step2: workshop; Step3: black leather chair; Step4: black leather chair.\nNew observation descriptions: Step1: a modern kitchen with stainless steel appliances and a large island. Step2: the foyer of a large house with high ceilings, windows, doors and a fireplace. Step3: the interior of a modern house with large windows, a living room and kitchen. Step4: a dining table and chairs in a modern kitchen and dining area of an apartment.\nOriginal Instruction: Leave the workshop and head to the black leather chair. Stop next the chair and the red wicker chairs.\nRewritten instruction: Go out of the kitchen and walk forward to the large windows. Stand next a dining table and chairs.\n
            Here is the instruction you need to rewrite.\n
            Landmarks in original trajectory: {objs}\n
            New observation descriptions: {sequential_scene_description}\n
            Original instruction: {instructions[cn-1]}
            """
            temp_instr.append(get_query_gpt35(prompt_4))
        temp_dic['original_instructions'] = instructions
        temp_dic["new_instrctions"] = temp_instr
        with open(output_file_path, 'w') as f2:
            json.dump(temp_dic, f2, indent=4)
    print('finished')



def r2r_instr_gen_step2():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    r2r_instr_path = '' # you can get from function 'r2r_instr_gen_step1'
    r2r_train_path = '' # original train json file
    output_path = ''

    with open(r2r_train_path, 'r') as fre:
        r2r_train_data = json.load(fre)
    output_new_train_data = []
    paths_list = os.listdir(r2r_instr_path)
    for p in tqdm(paths_list):
        file_path = os.path.join(r2r_instr_path, p)
        with open(file_path, 'r') as fpi:
            instr_data = json.load(fpi)
        path_id = instr_data['path_id']
        for i in range(len(r2r_train_data)):
            if path_id == r2r_train_data[i]['path_id']:
                for j in range(3):
                    temp_data = copy.deepcopy(r2r_train_data[i])
                    temp_data['path_id'] = str(temp_data['path_id']) + '_rwt_' + str(j+1)
                    if 'Rewritten instruction: ' in instr_data['new_instrctions'][j]:
                        new_instruction = instr_data['new_instrctions'][j].split('Rewritten instruction: ')[1]
                    if 'Rewritten Instruction: ' in instr_data['new_instrctions'][j]:
                        new_instruction = instr_data['new_instrctions'][j].split('Rewritten Instruction: ')[1]
                    temp_data['instructions'] = [new_instruction]
                    batch = tokenizer(temp_data['instructions'], padding=True, truncation=True, return_tensors="pt")
                    input_ids = batch['input_ids']
                    mask = batch['attention_mask'].eq(1)
                    input_ids = [input_ids[i][mask[i]].tolist() for i in range(len(mask))]
                    temp_data['instr_encodings'] = input_ids
                    output_new_train_data.append(temp_data)

    with open(output_path,'w') as f3:
        json.dump(output_new_train_data, f3, indent=4)



def reverie_instr_gen_step1():
    reverie_train_path = '' # the original train json file
    reverie_description_path = '' # description json file
    output_path = '' # specify your output path

    with open(reverie_train_path, 'r') as f1:
        reverie_train_data = json.load(f1)
    with open(reverie_description_path, 'r') as f2:
        reverie_description_data = json.load(f2)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    sampled_files = reverie_train_data

    for i in tqdm(range(len(sampled_files))):
        temp_dic = {}
        path_id = sampled_files[i]['path_id']
        all_id = sampled_files[i]['id']
        instructions = sampled_files[i]['instructions']
        landmark_list = sampled_files[i]['landmark_list']
        output_file_path = os.path.join(output_path, 'id_'+str(all_id)+'.json')
        if os.path.exists(output_file_path):
            continue
        temp_dic['path_id'] = path_id
        temp_dic['id'] = all_id
        temp_instr = []
        new_gt = {}
        for cn in range(1,4,1):
            gt_key = str(path_id) + '_rwt_' + str(cn)
            gt_actions = r2r_description_data[gt_key]
            new_gt[gt_key] = gt_actions
            if len(instructions) == 1:
                instructions = [instructions[0],instructions[0],instructions[0]]
                landmark_list[str(path_id)+'_1'] = landmark_list[str(path_id)+'_0']
                landmark_list[str(path_id)+'_2'] = landmark_list[str(path_id)+'_0']
            elif len(instructions) == 2:
                instructions = [instructions[0],instructions[1],instructions[0]]
                landmark_list[str(path_id)+'_2'] = landmark_list[str(path_id)+'_0']
            sequential_scene_description = ''
            objs = ''
            for k, v in gt_actions.items():
                step_idx = k.split('$')[0]
                sequential_scene_description += 'Step '+step_idx+':'+v+'.\t'  
            for k, v in landmark_list[str(path_id)+'_'+str(cn-1)].items():
                objs += 'Step '+k+':'+v+'.\t' 
            prompt_4 = f"""Rewrite the original instruction according to the observation descriptions given. The rewritten instruction should satisfy the following principles: 1) The rewritten instruction should replace the object (scene) described in the given instruction with a specific object (scene) mentioned in the new trajectory information according to the matched step information. The rewritten instruction should not contain the objects mentioned in the given instruction. 2) The rewritten instruction must have similar length with the given instruction. 3) The rewritten instruction must not contain the word "Step". 4) The rewritten instruction should change synonyms of verbs in the original instruction, like changing "turn left" to "take a left to". \nHere is an example.\nLandmarks in original trajectory: Step1: workshop; Step2: workshop; Step3: black leather chair; Step4: black leather chair.\nNew observation descriptions: Step1: a modern kitchen with stainless steel appliances and a large island. Step2: the foyer of a large house with high ceilings, windows, doors and a fireplace. Step3: the interior of a modern house with large windows, a living room and kitchen. Step4: a dining table and chairs in a modern kitchen and dining area of an apartment.\nOriginal Instruction: Leave the workshop and head to the black leather chair. Stop next the chair and the red wicker chairs.\nRewritten instruction: Go out of the kitchen and walk forward to the large windows. Stand next a dining table and chairs.\n
            Here is the instruction you need to rewrite.\n
            Landmarks in original trajectory: {objs}\n
            New observation descriptions: {sequential_scene_description}\n
            Original instruction: {instructions[cn-1]}
            """
            temp_instr.append(get_query_gpt35(prompt_4))
        temp_dic['original_instructions'] = instructions
        temp_dic["new_instrctions"] = temp_instr
        temp_dic['original_gt'] = landmark_list
        temp_dic['new_gt'] = new_gt
        with open(output_file_path, 'w') as f2:
            json.dump(temp_dic, f2, indent=4)
        # break
    print('finished')



def reverie_instr_gen_step2():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    reverie_instr_path = '' # reverie_instr_gen_step1
    reverie_train_path = '' # original train json file
    output_path = ''

    with open(reverie_train_path, 'r') as f1:
        reverie_train_data = json.load(f1)
    output_new_train_data = []
    paths_list = os.listdir(reverie_instr_path)
    for p in tqdm(paths_list):
        file_path = os.path.join(reverie_instr_path, p)
        with open(file_path, 'r') as f2:
            instr_data = json.load(f2)
        path_obj_id = instr_data['id']
        path_id = instr_data['path_id']
        for i in range(len(reverie_train_data)):
            if path_obj_id == reverie_train_data[i]['id']:
                for j in range(3):
                    temp_data = copy.deepcopy(reverie_train_data[i])
                    temp_data['id'] = str(temp_data['id']) + '_rwt_' + str(j+1)
                    temp_data['path_id'] = str(temp_data['path_id']) + '_rwt_' + str(j+1)
                    if 'Rewritten instruction: ' in instr_data['new_instrctions'][j]:
                        new_instruction = instr_data['new_instrctions'][j].split('Rewritten instruction: ')[1]
                    if 'Rewritten Instruction: ' in instr_data['new_instrctions'][j]:
                        new_instruction = instr_data['new_instrctions'][j].split('Rewritten Instruction: ')[1]
                    temp_data['instructions'] = [new_instruction]
                    batch = tokenizer(temp_data['instructions'], padding=True, truncation=True, return_tensors="pt")
                    input_ids = batch['input_ids']
                    mask = batch['attention_mask'].eq(1)
                    input_ids = [input_ids[i][mask[i]].tolist() for i in range(len(mask))]
                    temp_data['instr_encodings'] = input_ids
                    output_new_train_data.append(temp_data)

    with open(output_path,'w') as f3:
        json.dump(output_new_train_data, f3, indent=4)





if __name__ == "__main__":
    r2r_instr_gen_step1()
    r2r_instr_gen_step2()
    reverie_instr_gen_step1()
    reverie_instr_gen_step2()










