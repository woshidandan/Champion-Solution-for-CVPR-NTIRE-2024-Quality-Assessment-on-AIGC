import torch
import numpy as np
import clip
import random
from utils import set_dataset_aigc, _preprocess4
import torch.nn.functional as F
from itertools import product
import os
import tqdm

preprocess4 = _preprocess4()

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
aigc_set = './data/AIGCQA-30K-Image/test/'
seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def do_batch_prompt(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, _ = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    logits_per_image = torch.cat((logits_per_image[:,:num_patch-1,:].mean(1).unsqueeze(1), logits_per_image[:,-1,:].unsqueeze(1)), 1)

    logits_per_image_p = logits_per_image[0,:,:5].unsqueeze(0)
    if batch_size > 1:
        for i in range(1, batch_size):
            logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,:,i*5:(i+1)*5].unsqueeze(0)), 0)
    # logits_per_text = logits_per_text.mean(2)

    logits_per_image_p = F.softmax(logits_per_image_p, dim=2)

    return logits_per_image_p.mean(1)

qualitys_p = ['badly', 'poorly', 'fairly', 'well', 'perfectly']

def eval(loader, phase, dataset):
    model.eval()
    q_hat = {}
    with torch.cuda.amp.autocast(enabled=True):
        for sample_batched in tqdm.tqdm(loader):

            x, gmos, prompt,image_name= sample_batched['I'], sample_batched['mos'], sample_batched['prompt'], sample_batched['image_name']

            x = x.to(device)
            texts = [f"a photo that {c} matches '{p}'" for p,c in product(prompt, qualitys_p)]
            input_texts = torch.cat([clip.tokenize(c,truncate=True) for c in texts]).to(device)   

            with torch.no_grad():
                logits_per_image = do_batch_prompt(x, input_texts)

            logits_quality = logits_per_image
            quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                                4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            quality_preds = (quality_preds-1)/4*5
            # 将分数以及对应的图片名字存储到字典中
            for i in range(len(image_name)):
                q_hat[image_name[i]] = quality_preds[i].item()

        print_text = dataset + ' ' + phase + ' finished'
        print(print_text)

        return  q_hat


num_workers = 8

model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

ckpt = './checkpoints_AIGCQA30K_vit14l/1/quality_best_ckpt.pt'
checkpoint = torch.load(ckpt)

model.load_state_dict(checkpoint['model_state_dict'])

aigc_test_csv = os.path.join('./data/AIGCQA-30K-Image/','info_test.csv')

aigc_test_loader = set_dataset_aigc(aigc_test_csv, 16, aigc_set, num_workers, preprocess4, 15, True, True)

q_hat1 = eval(aigc_test_loader, 'test', 'live')

# 将字典中的图片名以及对应分数存入txt文件中，每对占一行，用逗号分隔
with open('output.txt', 'w') as f:
    for k, v in q_hat1.items():
        f.write(k + ',' + str(v) + '\n')