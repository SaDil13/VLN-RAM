
import json
import os
import json
from diffusers import AutoPipelineForText2Image
import torch
import requests
from tqdm import tqdm
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import argparse
num_inference_steps_gpt35 = 30

def load_model():
    import torch
    from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler
    # model_ckpt = "stabilityai/stable-diffusion-2-base"
    model_ckpt = "stabilityai/stable-diffusion-2-1-base"
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
    pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


def get_panorama(pipe, prompt='a picture of mountains', num_infer_steps=30):
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    prompt_embeds = compel(prompt)
    image = pipe(prompt_embeds=prompt_embeds, num_inference_steps=num_infer_steps).images[0]
    return image


def get_descriptions_to_panoramas(start, interval):
    pipe = load_model()
    out_basic_path = '' # specify your output path 
    root_path = '' # put the description json file here
    if not os.path.exists(out_basic_path):
        os.makedirs(out_basic_path, exist_ok=True)
    
    with open(root_path, 'r') as rr:
        description_data = json.load(rr)
    dealwith_data = description_data
    for i in tqdm(range(len(dealwith_data))):
        path_id = dealwith_data[i]['path_id']
        scan = dealwith_data[i]['scan']
        paths = dealwith_data[i]['path']
        new_descriptions = dealwith_data[i]['new_descriptions']
        output_path = os.path.join(out_basic_path, 'pathid_'+str(path_id))
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        if len(os.listdir(output_path)) == len(paths):
            continue
        for p in range(len(paths)):
            pano_name = '%s_%s' % (scan, paths[p])
            new_descriptions_part = new_descriptions[pano_name] 
            final_output_path = os.path.join(output_path, str(p)+'$'+pano_name)
            image_1 = get_panorama(prompt=new_descriptions_part[0], num_infer_steps=num_inference_steps_gpt35, pipe=pipe)
            image_2 = get_panorama(prompt=new_descriptions_part[1], num_infer_steps=num_inference_steps_gpt35, pipe=pipe)
            image_3 = get_panorama(prompt=new_descriptions_part[2], num_infer_steps=num_inference_steps_gpt35, pipe=pipe)
            if not os.path.exists(final_output_path):
                os.makedirs(final_output_path, exist_ok=True)
            save_name_1 = 'description_1.png'
            image_1.save(os.path.join(final_output_path, save_name_1))
            save_name_2 = 'description_2.png'
            image_2.save(os.path.join(final_output_path, save_name_2))
            save_name_3 = 'description_3.png'
            image_3.save(os.path.join(final_output_path, save_name_3))



def get_panorama_discretized(path, output_path):
    import os
    import cv2 
    import Equirec2Perspec as E2P 
    import matplotlib.pyplot as plt
    from PIL import Image

    id2angle = {
        0:(0,-30), 1:(30,-30), 2:(60,-30), 3:(90,-30), 4:(120,-30), 5:(150,-30),
        6:(180,-30), 7:(210,-30), 8:(240,-30), 9:(270,-30), 10:(300,-30), 11:(330,-30), 
        12:(0,0), 13:(30,0), 14:(60,0), 15:(90,0), 16:(120,0), 17:(150,0),
        18:(180,0), 19:(210,0), 20:(240,0), 21:(270,0), 22:(300,0), 23:(330,0), 
        24:(0,30), 25:(30,30), 26:(60,30), 27:(90,30), 28:(120,30), 29:(150,30),
        30:(180,30), 31:(210,30), 32:(240,30), 33:(270,30), 34:(300,30), 35:(330,30), 
    }
    # path like: pathid_32/0$1LXtFkjw3qL_3c8c9d438bdd4ebaad5a0d0e93a43c0f
    for root, dirs, files in os.walk(path):
        for f in files:
            pic_path = os.path.join(root, f)
            equ = E2P.Equirectangular(pic_path)    # Load equirectangular image
            for i in range(36):
                (theta, phi) = id2angle[i]
                img = equ.GetPerspective(60, theta, phi, 480, 640) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                pic_output_dic_path = os.path.join(output_path, f.split('.png')[0])
                if not os.path.exists(pic_output_dic_path):
                    os.makedirs(pic_output_dic_path)
                pic_output_path = os.path.join(pic_output_dic_path, str(i)+'.jpg')
                pil_img.save(pic_output_path)


def get_pano_to_single(start, interval):
    panorama_gpt35_path = ''
    output_path = ''
    paths_list = os.listdir(panorama_gpt35_path)
    
    if start + interval >= len(paths_list):
        sampled_files = paths_list[start:]
    else:
        sampled_files = paths_list[start:start+interval]
    for pathid in tqdm(sampled_files):
        path_data_path = os.path.join(panorama_gpt35_path, pathid)
        output_data_path = os.path.join(output_path, pathid)
        if os.path.exists(output_data_path):
            if len(os.listdir(path_data_path)) == len(os.listdir(output_data_path)):
                continue
        for step in os.listdir(path_data_path):
            vp_data_path = os.path.join(path_data_path, step)
            output_vp_data_path = os.path.join(output_data_path, step)
            get_panorama_discretized(path=vp_data_path, output_path=output_vp_data_path)


if __name__ == '__main__':
    get_descriptions_to_panoramas()
    get_pano_to_single()

