import torch
import numpy as np
import clip
import random
from utils import set_dataset_aigc, _preprocess2
import torch.nn.functional as F
from itertools import product
import os
import tqdm

preprocess2 = _preprocess2()

aigc_set = './data/AIGCQA-30K-Image/test/'
aigc_test_csv = './data/AIGCQA-30K-Image/info_test.csv'

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def do_batch_prompt(model,x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, _ = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image, logits_per_image_r = logits_per_image.split(num_patch-1, dim=1)
    logits_per_image = logits_per_image.mean(1)

    logits_per_image_p = logits_per_image[0,:5].unsqueeze(0)
    if batch_size > 1:
        for i in range(1, batch_size):
            logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,i*5:(i+1)*5].unsqueeze(0)), 0)

    logits_per_image_p = F.softmax(logits_per_image_p, dim=1)

    logits_per_image_r = logits_per_image_r.squeeze(1)
    logits_per_image_r_p = logits_per_image_r[0,:5].unsqueeze(0)
    if batch_size > 1:
        for i in range(1, batch_size):
            logits_per_image_r_p = torch.cat((logits_per_image_r_p, logits_per_image_r[i,i*5:(i+1)*5].unsqueeze(0)), 0)
    logits_per_image_r_p = F.softmax(logits_per_image_r_p, dim=1)
    logits_per_image = (logits_per_image_p + logits_per_image_r_p) / 2
    logits_quality = logits_per_image
    quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
    quality_preds = (quality_preds-1)/4*5
    return quality_preds


qualitys_p = ['badly', 'poorly', 'fairly', 'well', 'perfectly']

def eval(loader, phase, dataset):
    model.eval()
    model2.eval()
    q_hat = {}
    with torch.cuda.amp.autocast(enabled=True):
        for sample_batched in tqdm.tqdm(loader):

            x, gmos, prompt, prompt_name, image_name= sample_batched['I'], sample_batched['mos'], sample_batched['prompt'], sample_batched['prompt_name'], sample_batched['image_name']

            x = x.to(device)
            texts = [f"a photo that {c} matches '{p}'" for p,c in product(prompt, qualitys_p)]
            input_texts = torch.cat([clip.tokenize(c,truncate=True) for c in texts]).to(device)   
            texts2 = [f"a photo that {c} matches '{p}'" for p,c in product(prompt_name, qualitys_p)]
            input_texts2 = torch.cat([clip.tokenize(c,truncate=True) for c in texts2]).to(device) 
            
            # Calculate features
            with torch.no_grad():
                quality_preds = do_batch_prompt(model, x, input_texts)
                quality_preds2 = do_batch_prompt(model2, x, input_texts2)

            # logits_quality = logits_per_image
            # quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
            #                     4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            # quality_preds = (quality_preds-1)/4*5

            for i in range(len(image_name)):
                q_hat[image_name[i]] = (quality_preds[i].item() * quality_preds2[i].item())/5

        print_text = dataset + ' ' + phase + ' finished'
        print(print_text)

        return  q_hat


num_workers = 8

model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
ckpt = './weight/nomodelname.pth'
checkpoint = torch.load(ckpt)

model.load_state_dict(checkpoint)


model2, preprocess = clip.load("ViT-L/14", device=device, jit=False)
ckpt2 = './weight/modelname.pth'
checkpoint2 = torch.load(ckpt2)

model2.load_state_dict(checkpoint2)

aigc_test_loader = set_dataset_aigc(aigc_test_csv, 16, aigc_set, num_workers, preprocess2, 15, True, True)

q_hat1 = eval(aigc_test_loader, 'test', 'live')

with open('output1.txt', 'w') as f:
    for k, v in q_hat1.items():
        f.write(k + ',' + str(v) + '\n')


