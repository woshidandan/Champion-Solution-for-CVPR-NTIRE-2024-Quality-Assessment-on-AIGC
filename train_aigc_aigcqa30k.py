import torch
import numpy as np
#from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import clip
import random
from MNL_Loss import loss_m3
import scipy.stats
from utils import set_dataset_aigc, _preprocess4, _preprocess5, convert_models_to_fp32
import torch.nn.functional as F
from itertools import product
import os
import tqdm


checkpoint_dir = 'checkpoints_AIGCQA30K_vit14l'

qualitys_p = ['badly', 'poorly', 'fairly', 'well', 'perfectly']

aigc_set  = './data/AIGCQA-30K-Image/train/'

seed = 20200626

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

initial_lr = 5e-6
num_epoch = 80
bs = 32
train_patch = 8


preprocess4 = _preprocess4()
preprocess5 = _preprocess5()

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

# 整合相似度结果
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
        model.train()
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
            # print(total_loss)
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)


            running_loss += total_loss.data.item()
            avg_loss = running_loss / (step + 1)
            # print(avg_loss)
            loop.set_description('Epoch:{}  Loss:{:.4f}'.format(epoch, avg_loss))

        if (epoch >= 0):
            srcc1 = eval(live_val_loader, phase='val', dataset='live')

            if srcc1 > best_result['quality']:
                print('**********New quality best!**********')
                best_epoch['quality'] = epoch
                best_result['quality'] = srcc1
                # srcc_dict1['live'] = srcc11
                dir = os.path.join(checkpoint_dir, str(session + 1))
                os.makedirs(dir,exist_ok = True)
                ckpt_name = os.path.join(checkpoint_dir, str(session + 1), 'quality_best_ckpt.pt')
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'all_results': all_result
                }, ckpt_name)  # just change to your preferred folder/filename

            # if epoch in[3,4,5,6]:
            #     dir = os.path.join(checkpoint_dir, str(session + 1))
            #     os.makedirs(dir,exist_ok = True)
            #     ckpt_name = os.path.join(checkpoint_dir, str(session + 1), 'ckpt_{}.pt'.format(epoch))
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         # 'all_results': all_result
            #     }, ckpt_name)


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
    print(print_text)

    return  (srcc+plcc)/2


num_workers = 8
for session in range(0,10):

    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr,
        weight_decay=0.001)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)


    train_loss = []
    start_epoch = 0

    freeze_model(opt)

    best_result = {'avg': 0.0, 'quality': 0.0}
    best_epoch = {'avg': 0, 'quality': 0}

    live_train_csv = os.path.join('./Database/AIGC', str(session+1), 'train.csv')
    live_val_csv = os.path.join('./Database/AIGC', str(session+1), 'val.csv')
    all_train_csv = 'data/AIGCQA-30K-Image/info_train.csv'
    # live_test_csv = os.path.join('./IQA_Database/AIGC', str(session+1), 'live_test_clip.txt')

    live_train_loader = set_dataset_aigc(live_train_csv, 3, aigc_set, num_workers, preprocess4, train_patch, False)
    live_val_loader = set_dataset_aigc(live_val_csv,16, aigc_set, num_workers, preprocess5, 15, True)
    # live_test_loader = set_dataset(live_test_csv, 16, live_set, num_workers, preprocess2, 15, True)

    train_loaders = [live_train_loader]

    result_pkl = {}
    for epoch in range(0, num_epoch):
        # best_result, best_epoch, srcc_dict, all_result = train(model, best_result, best_epoch, srcc_dict)
        best_result, best_epoch = train(model, best_result, best_epoch)
        scheduler.step()

        print('...............current quality best...............')
        print('best quality epoch:{}'.format(best_epoch['quality']))
        print('best quality result:{}'.format(best_result['quality']))