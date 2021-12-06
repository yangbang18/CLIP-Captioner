import sys
sys.path.append('..')
sys.path.append('.')
import torch
from CLIP import clip
import os
import argparse
import pickle
import h5py
import glob
from config import Constants
from PIL import Image
from tqdm import tqdm
from misc.utils import get_uniform_items_from_k_snippets
from dataloader import get_ids_set


def prepare_encoded_image_feats(model, preprocess, device, all_frames_path, frames_suffix, video_ids, db):
    # the original weights of CLIP have the type of torch.float16 (model.half())
    # changing it to torch.float32 (model.float()) is important to get the identical captioning performance
    model.float()
    model.eval()
    model.to(device)
    
    for _id in tqdm(video_ids):
        vid = 'video{}'.format(_id)
        if db is not None and vid in db.keys(): 
            continue

        frames = sorted(glob.glob(os.path.join(all_frames_path, vid, '*.{}'.format(frames_suffix))))
        frames = get_uniform_items_from_k_snippets(frames, Constants.n_total_frames) # uniformly sampling 60 frames
        images_of_this_vid = [preprocess(Image.open(f)) for f in frames] # preprocess and transform these sampled frames
        images_of_this_vid = torch.stack(images_of_this_vid, dim=0).to(device)

        with torch.no_grad():
            image_feats_of_this_vid = model.encode_image(images_of_this_vid)
            if image_feats_of_this_vid.dim() > 2:
                image_feats_of_this_vid = image_feats_of_this_vid.squeeze()

            print(vid, image_feats_of_this_vid.shape)
            db[vid] = image_feats_of_this_vid.cpu().numpy()


def get_root(args):
    return os.path.join(Constants.base_data_path, args.dataset)

def get_feats_save_path(args, root=None):
    if root is None:
        root = get_root(args)
    
    args.arch = args.arch.replace('/', '-')
    feats_save_path = os.path.join(root, 'feats')
    os.makedirs(feats_save_path, exist_ok=True)

    if args.replace:
        feats_save_path = os.path.join(feats_save_path, 'CLIP_{}_rep.hdf5'.format(args.arch))
    else:
        feats_save_path = os.path.join(feats_save_path, 'CLIP_{}.hdf5'.format(args.arch))

    return feats_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'MSVD', 'VATEX'])
    parser.add_argument('-fp', '--all_frames_path', type=str, default='')
    parser.add_argument('-fs', '--frames_suffix', type=str, default='jpg')
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('-replace', '--replace', default=False, action='store_true')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)
    if args.replace:
        model.visual.attnpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    if not args.all_frames_path:
        args.all_frames_path = os.path.join(Constants.base_data_path, args.dataset, 'all_frames')
    assert os.path.exists(args.all_frames_path)
    
    root = os.path.join(Constants.base_data_path, args.dataset)
    
    info_corpus_path = os.path.join(root, 'info_corpus.pkl')
    split = pickle.load(open(info_corpus_path, 'rb'))['info']['split']
    all_video_ids = get_ids_set('all', split, is_vatex_activate=(args.dataset=='VATEX'))

    feats_save_path = get_feats_save_path(args, root)
    print('- Save all feats to {}'.format(feats_save_path))
    
    db = h5py.File(feats_save_path, 'a')
    prepare_encoded_image_feats(model, preprocess, device, args.all_frames_path, args.frames_suffix, all_video_ids, db)
    db.close()

