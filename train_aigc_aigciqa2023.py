import torch
import numpy as np
from torch.optim import lr_scheduler
import clip
import random
from MNL_Loss import loss_m3
import scipy.stats
from utils import set_dataset_aigc_2023, _preprocess2, _preprocess3, convert_models_to_fp32, get_logger,log_and_print 
import torch.nn.functional as F
from itertools import product
import os
import tqdm


checkpoint_dir = 'checkpoints_AIGCIQA023_vit32b'
os.makedirs(checkpoint_dir,exist_ok = True)

qualitys_p = ['badly', 'poorly', 'fairly', 'well', 'perfectly']

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 5e-6
num_epoch = 15
bs = 32
train_patch = 8


preprocess2 = _preprocess2()
preprocess3 = _preprocess3()

opt = 0
def freeze_model(opt):
    model.logit_scale.requires_grad = False
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in model.token_embedding.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = False
        model.positional_embedding.requires_grad = False
        model.text_projection.requires_grad = False
        for p in model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in model.parameters():
            p.requires_grad =False


def do_batch_prompt1(x, text):
    batch_size = x.size(0)
    num_patch = x.size(1)

    x = x.view(-1, x.size(2), x.size(3), x.size(4))

    logits_per_image, _ = model.forward(x, text)

    logits_per_image = logits_per_image.view(batch_size, num_patch, -1)
    # logits_per_text = logits_per_text.view(-1, batch_size, num_patch)

    logits_per_image, logits_per_image_r = logits_per_image.split(num_patch-1, dim=1)

    logits_per_image = logits_per_image.mean(1)

    logits_per_image_p = logits_per_image[0,:5].unsqueeze(0)
    if batch_size > 1:
        for i in range(1, batch_size):
            logits_per_image_p = torch.cat((logits_per_image_p, logits_per_image[i,i*5:(i+1)*5].unsqueeze(0)), 0)
    # logits_per_text = logits_per_text.mean(2)

    logits_per_image_p = F.softmax(logits_per_image_p, dim=1)


    logits_per_image_r = logits_per_image_r.squeeze(1)
    logits_per_image_r_p = logits_per_image_r[0,:5].unsqueeze(0)
    if batch_size > 1:
        for i in range(1, batch_size):
            logits_per_image_r_p = torch.cat((logits_per_image_r_p, logits_per_image_r[i,i*5:(i+1)*5].unsqueeze(0)), 0)
    logits_per_image_r_p = F.softmax(logits_per_image_r_p, dim=1)

    return (logits_per_image_p + logits_per_image_r_p) / 2


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


def train(model, best_result, best_epoch):
    with torch.cuda.amp.autocast(enabled=True):

        running_loss = 0
        model.eval()
        loader = train_loaders[0]

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
            scheduler.step()
            print(optimizer.state_dict()['param_groups'][0]['lr'])

        avg_loss = 0
        step = -1
        loop = tqdm.tqdm(loader, desc='Epoch:{}'.format(epoch))
        for sample_batched in loop:

            step += 1
            x, gmos, prompt, image_name = sample_batched['I'], sample_batched['mos'],  sample_batched['prompt'], sample_batched['image_name']
            x = x.to(device)
            gmos = gmos.to(device)
            texts = [f"a photo that {c} matches '{p}'" for p,c in product(prompt, qualitys_p)]
            input_texts = torch.cat([clip.tokenize(c,truncate=True) for c in texts]).to(device)  

            optimizer.zero_grad()
            logits_quality= do_batch_prompt(x, input_texts)

            logits_quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                                4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            logits_quality = ((logits_quality -1) / 4) *5

            total_loss = loss_m3(logits_quality, gmos.detach()).mean()
            total_loss = total_loss
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)


            running_loss += total_loss.data.item()
            avg_loss = running_loss / (step + 1)
            loop.set_description('Epoch:{}  Loss:{:.4f}'.format(epoch, avg_loss))

        if (epoch >= 0):
            avg_score,srcc,plcc = eval(aigc_val_loader, phase='val', dataset='live')

            if avg_score > best_result['quality']:
                log_and_print(base_logger,'**********New quality best!**********')
                best_epoch['quality'] = epoch
                best_result['quality'] = avg_score
                best_result['srcc'] = srcc
                best_result['plcc'] = plcc
                dir = os.path.join(checkpoint_dir, str(session + 1))
                os.makedirs(dir,exist_ok = True)
                ckpt_name = os.path.join(checkpoint_dir, str(session + 1), 'quality_best_ckpt.pt')
                
                torch.save({
                    # 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'all_results': all_result
                }, ckpt_name)  # just change to your preferred folder/filename

        # return best_result, best_epoch, srcc_dict, all_result
        return best_result, best_epoch


def eval(loader, phase, dataset):
    model.eval()

    q_mos = []
    q_hat = []

    for sample_batched in tqdm.tqdm(loader, desc='{}:{}'.format(dataset, phase)):

        x, gmos, prompt, image_name = sample_batched['I'], sample_batched['mos'],  sample_batched['prompt'], sample_batched['image_name']

        x = x.to(device)
        q_mos = q_mos + gmos.cpu().tolist()
        texts = [f"a photo that {c} matches '{p}'" for p,c in product(prompt, qualitys_p)]
        input_texts = torch.cat([clip.tokenize(c,truncate=True) for c in texts]).to(device)  

        with torch.no_grad():
            logits_quality = do_batch_prompt(x, input_texts)

        quality_preds = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
        quality_preds = ((quality_preds -1) / 4) *5
        q_hat = q_hat + quality_preds.cpu().tolist()


    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    #计算PLCC
    plcc = scipy.stats.pearsonr(x=q_mos, y=q_hat)[0]

    print_text = dataset + ':' + phase + ': ' +  'srcc:{:.4f}  plcc{:.4f}'.format(srcc,plcc)
    # print_text = dataset + ' ' + phase + ' finished'
    log_and_print(base_logger,print_text)

    return  (srcc+plcc)/2, srcc, plcc



num_workers = 16
best_result_list = []
base_logger = get_logger(os.path.join(checkpoint_dir,'train_test.log'), 'log')
for session in range(0,10):

    # model, preprocess = clip.load("RN50", device=device, jit=False)
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    # model, preprocess = clip.load("ViT-L/14", device=device, jit=False)    
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    train_loss = []
    start_epoch = 0

    freeze_model(opt)

    best_result = { 'quality': 0.0 ,'srcc': 0.0, 'plcc': 0.0}
    best_epoch = {  'quality': 0}

    aigc_train_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'train.csv')
    aigc_val_csv = os.path.join('./Database/AIGCIQA2023', str(session+1), 'val.csv')

    aigc_set = './data/AIGCIQA2023'
    aigc_train_loader = set_dataset_aigc_2023(aigc_train_csv, 16, aigc_set, num_workers, preprocess3, 8, False)
    aigc_val_loader = set_dataset_aigc_2023(aigc_val_csv,16, aigc_set, num_workers, preprocess2,15, True)
    # live_test_loader = set_dataset(live_test_csv, 16, live_set, num_workers, preprocess2, 15, True)

    train_loaders = [aigc_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        best_result, best_epoch = train(model, best_result, best_epoch)
        scheduler.step()
        log_and_print(base_logger,'...............current quality best, session:{}...............'.format(session + 1))
        log_and_print(base_logger,'best quality epoch:{}'.format(best_epoch['quality']))
        log_and_print(base_logger,'best quality result:{}, srcc:{}, plcc{}'.format(best_result['quality'], best_result['srcc'], best_result['plcc']))

    best_result_list.append(best_result)

avg_srcc = 0
avg_plcc = 0
for i in range(10):
    avg_srcc += best_result_list[i]['srcc']
    avg_plcc += best_result_list[i]['plcc']

avg_srcc = avg_srcc / 10
avg_plcc = avg_plcc / 10
log_and_print(base_logger,'all_finished,average srcc:{}, plcc:{}'.format(avg_srcc, avg_plcc))