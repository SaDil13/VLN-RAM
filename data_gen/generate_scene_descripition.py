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



def sumup_descriptions():
    dir_path = '' # where you get the output dir by descriptions_generate function
    panorama_tag2text = '' # put your panorama json extracted by tag2text model
    with open(panorama_tag2text, 'r') as pr:
        panorama_data = json.load(pr)
    temp_list = []
    error_list = []
    print(len(os.listdir(dir_path)))
    return
    for f in os.listdir(dir_path):
        temp_dict = {}
        file_path = os.path.join(dir_path, f)
        with open(file_path, 'r', encoding='utf-8') as ff:
            json_data = json.load(ff)
        try:
            for k, v in json_data['new_descriptions'].items():
                temp_1 = v.split('<1>')[1].split('<2>')[0].strip()
                temp_list_1 = temp_1.split('Rewritten descriptions: ')[1]
                temp_2 = v.split('<2>')[1].split('<3>')[0].strip()
                temp_list_2 = temp_2.split('Rewritten descriptions: ')[1]
                temp_3 = v.split('<3>')[1].strip()
                temp_list_3 = temp_3.split('Rewritten descriptions: ')[1]

                json_data['new_descriptions'][k] = [temp_list_1, temp_list_2, temp_list_3]

            temp_list.append(json_data)
        except:
            error_list.append(f)
    output_path = ''
    with open(output_path, 'w', encoding='utf-8') as fo:
        json.dump(temp_list, fo, indent=4)



def descriptions_generate(start, interval): 
    r2r_train_path = '' # original train json file
    output_path = '' # specify your output path
    panorama_tag2text_filepath = '' # panorama json extracted by tag2text
    with open(panorama_tag2text_filepath, 'r') as fr:
        description_data = json.load(fr)
    # if you get the demonstrations by chatgpt, you can specify this, or you just ignore it.
    # description_demonstrations_filepath = '' 
    # with open(description_demonstrations_filepath, 'r') as fr2:
    #     description_demonstrations_data = json.load(fr2)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    with open(r2r_train_path, 'r') as fr:
        data = json.load(fr)
    sampled_files = data
    # print(len(sampled_files))
    for fn in tqdm(sampled_files):
        flag = False
        instructions = fn['instructions'] # a list of 3 strings
        temp_dict = {}
        temp_dict['path_id'] = fn['path_id']
        output_file_path = os.path.join(output_path, 'pathid_'+str(fn['path_id'])+'.json')
        if os.path.exists(output_file_path):
            print('skip')
            continue
        temp_dict['scan'] = fn['scan']  
        temp_dict['path'] = fn['path'] # just sample 1st vp
        temp_list = {}
        #### you can uncomment this to introduce demonstrations ####
        # random.seed(fn['path_id'])
        # random_key = random.choice(list(description_demonstrations_data.keys())) # a random pano_name
        # demon_descriptions = description_demonstrations_data[random_key]
        # example_description = description_data[random_key]
        # example_finegrained_description = ''
        # for d in range(len(demon_descriptions)):
        #     example_finegrained_description += "\n"+str(d+1)+" =>" + demon_descriptions[d]
        
        for pth in range(len(fn['path'])):
            pano_name = '%s_%s' % (fn['scan'], fn['path'][pth])
            description = description_data[pano_name]
            prompt_4 = f"""
            Please generate 3 rewritten descriptions for the description I provide to you. The principle is 1) You should rewrite the sentence pattern of description, 2) You should add the possible objects that may exist in the scene described in the given description. The sentence pattern and added objects should not be the same among 3 rewritten descriptions. You have to make sure that the added objects can be seen in the indoor environment of a building. \nYour output format should be: (You must output 3 times in the following format) \n<number(from 1 to 3)> Added objects: XXX. (at least 3 new objects)\nRewritten descriptions: XXX. (20 to 40 words)\nHere is an example.\n Scene description: a bathroom that has mirrors on the walls\n<1> Added objects: vanity, sink, faucet\nRewritten descriptions: A spacious bathroom featuring a large vanity with a marble countertop and double sinks, reflecting off the expansive mirrors on the walls, complemented by chrome faucets and a tiled backsplash.\nDescription you need to process: \n{description}
            """
            response = get_query_gpt35(prompt_4)
            temp_list[pano_name] = response

        temp_dict['new_descriptions'] = temp_list

        with open(output_file_path, 'w') as f2:
            json.dump(temp_dict, f2, indent=4)
    print('finished')


if __name__ == "__main__":
    descriptions_generate()











