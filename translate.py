import os
import torch

from models import ModelEnsemble, Model
from dataloader import get_loader
from pytorch_lightning import Trainer
import argparse
from tqdm import tqdm
from typing import Union

from misc.utils import to_device, save_dict_to_csv


def run_eval(
        args, 
        model: Union[ModelEnsemble, Model], 
        loader: torch.utils.data.DataLoader, 
        device: torch.device,
        return_details: bool = False,
    ):
    model.eval()
    model.to(device)
    
    vocab = model.get_vocab()

    all_step_outputs = []
    for batch in tqdm(loader):
        with torch.no_grad():
            for k in model.get_keys_to_device():
                if k in batch:
                    batch[k] = to_device(batch[k], device)

            step_outputs = model.translate_step(
                batch=batch,
                vocab=vocab,
                assert_only_a_caption_per_video=True,
                verbose=getattr(args, 'verbose', False),
            )
        all_step_outputs.append(step_outputs)
    
    scores, detail_scores, pred_captions = model.test_epoch_end(
        all_step_outputs=all_step_outputs,
        log_scores=False,
        verbose=True,
        save_csv_path=os.path.dirname(args.checkpoint_paths[0])
    )

    if return_details:
        return scores, detail_scores
    
    return scores


def loop_n_frames(args, model, device):
    opt = model.get_opt()
    for i in range(1, opt['n_frames']+1):
        loader = get_loader({**opt, 'n_frames': i}, 'test', print_info=True if i == 1 else False, specific=args.specific, 
            not_shuffle=True, is_validation=True
        )
        scores = run_eval(args, model, loader, device)
        scores['n_frames'] = i
        scores['scope'] = opt['scope']
        save_dict_to_csv('./results_loop/', "n_frames.csv", scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-cp', '--checkpoint_paths', type=str, nargs='+', required=True)

    cs = parser.add_argument_group(title='Common Settings')
    cs.add_argument('-gpus', '--gpus', type=int, default=1)
    cs.add_argument('-num_workers', '--num_workers', type=int, default=1)
    cs.add_argument('-fast', '--fast', default=False, action='store_true', 
        help='directly use Trainer.test()')
    cs.add_argument('-v', '--verbose', default=False, action='store_true',
        help='print some intermediate information (works when `fast` is False)')
    cs.add_argument('--save_csv', default=False, action='store_true',
        help='save result to csv file in model path (works when `fast` is False)')

    ds = parser.add_argument_group(title='Dataloader Settings')
    ds.add_argument('-bsz', '--batch_size', type=int, default=128)
    ds.add_argument('-mode', '--mode', type=str, default='test',
        help='which set to run?', choices=['train', 'validate', 'test', 'all'])
    ds.add_argument('-specific', '--specific', default=-1, type=int, 
        help='run on the data of the specific category (only works in the MSR-VTT)')

    ar = parser.add_argument_group(title='Autoregressive Decoding Settings')
    ar.add_argument('-bs', '--beam_size', type=int, default=5, help='Beam size')
    ar.add_argument('-ba', '--beam_alpha', type=float, default=1.0)
    ar.add_argument('-topk', '--topk', type=int, default=1)

    na = parser.add_argument_group(title='Non-Autoregressive Decoding Settings')
    na.add_argument('-i', '--iterations', type=int, default=5)
    na.add_argument('-lbs', '--length_beam_size', type=int, default=6)
    na.add_argument('-q', '--q', type=int, default=1)
    na.add_argument('-qi', '--q_iterations', type=int, default=1)
    na.add_argument('-paradigm', '--paradigm', type=str, default='mp', choices=['mp', 'ef', 'l2r'])
    na.add_argument('-use_ct', '--use_ct', default=False, action='store_true')
    na.add_argument('-md', '--masking_decision', default=False, action='store_true')
    na.add_argument('-ncd', '--no_candidate_decision', default=False, action='store_true')
    na.add_argument('--algorithm_print_sent', default=False, action='store_true')
    na.add_argument('--teacher_path', type=str, default='')

    ts = parser.add_argument_group(title='Task Settings')
    ts.add_argument('-latency', '--latency', default=False, action='store_true', 
        help='batch_size will be set to 1 to compute the latency, which will be saved to latency.txt in the checkpoint folder')
    
    parser.add_argument('-json_path', '--json_path', type=str, default='')
    parser.add_argument('-json_name', '--json_name', type=str, default='')
    parser.add_argument('-ns', '--no_score', default=False, action='store_true')
    parser.add_argument('-analyze', default=False, action='store_true')
    parser.add_argument('-collect_path', type=str, default='./collected_captions')
    parser.add_argument('-collect', default=False, action='store_true')
    parser.add_argument('-nobc', '--not_only_best_candidate', default=False, action='store_true')
            
    parser.add_argument('--attr', default=False, action='store_true')
    parser.add_argument('--probs_scaler', type=float, default=0.1)
    
    parser.add_argument('--with_backbones', type=str, nargs='+', default=[])
    parser.add_argument('--loop_n_frames', default=False, action='store_true')
    parser.add_argument('--calculate_mAP', default=False, action='store_true')

    args = parser.parse_args()

    if not args.teacher_path:
        del args.teacher_path

    if not args.with_backbones:
        del args.with_backbones
        strict = True
    else:
        strict = False
    
    if args.fast:
        # fast mode
        model = ModelEnsemble(args.checkpoint_paths, vars(args), strict=strict)
        trainer = Trainer(logger=False, gpus=args.gpus)
        opt = model.get_opt()
        loader = get_loader(opt, args.mode, print_info=True, specific=args.specific, 
            not_shuffle=True, batch_size=args.batch_size
        )
        trainer.test(model, loader)
    else:
        # handle the device and running loop on your own
        device = torch.device('cpu' if args.gpus == 0 else 'cuda')

        if len(args.checkpoint_paths) == 1:
            model = Model.load_from_checkpoint(
                args.checkpoint_paths[0], 
                new_opt_used_to_override=vars(args),
                map_location=device,
                strict=strict
            )
        else:
            # `load_from_checkpoint` is called in `ModelEnsemble.__init__()` for each checkpoint
            model = ModelEnsemble(args.checkpoint_paths, vars(args), map_location=device, strict=strict)
        print(model)
        
        if args.loop_n_frames:
            loop_n_frames(args, model, device)
        else:
            image_preprocess_func = None
            if hasattr(model.captioner, 'backbone') and model.captioner.backbone is not None:
                image_preprocess_func = model.captioner.backbone.get_preprocess_func('i')

            loader = get_loader(model.get_opt(), args.mode, print_info=True, specific=args.specific, 
                not_shuffle=True, batch_size=args.batch_size, is_validation=True, image_preprocess_func=image_preprocess_func, all_caps=getattr(args, 'all_caps', False)
            )
            run_eval(args, model, loader, device)
