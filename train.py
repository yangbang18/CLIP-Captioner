import warnings
warnings.filterwarnings('ignore')

from opts import get_opt
from models import Model
from dataloader import get_loader
from pytorch_lightning import (
    seed_everything, 
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':
    opt = get_opt()

    seed_everything(opt['seed'], workers=True)
    #seed_everything(1, workers=True)
    
    if opt.get('load_model_weights_from', ''):
        model = Model.load_from_checkpoint(
            opt['load_model_weights_from'],
            new_opt_used_to_override=opt,
            strict=opt.get('load_strictly', False),
            merge_opt=True,
        )

        names = set(opt.get('freeze_parameters_except', []))
        if len(names):
            print('- Parameter names that contain any string of {} are trainable'.format(names))
            for n, p in model.named_parameters():
                flag = sum([1 for specified_name in names if specified_name in n])
                if not flag:
                    p.requires_grad = False

            for n, p in model.named_parameters():
                if p.requires_grad: print(n)
        
        print('- Have loaded model weights from {}, strict: {}'.format(
            opt['load_model_weights_from'], opt.get('load_strictly', False)))    
    else:
        model = Model(opt)
    print(model)

    if opt['save_topk_models'] > 1:
        some_args_about_checkpoint = {
            'save_top_k': opt['save_topk_models'],
            'filename': 'E{epoch:02d}-B{Bleu_4:.3f}-M{METEOR:.3f}-R{ROUGE_L:.3f}-C{CIDEr:.3f}-Sum{Sum:.3f}',
            'auto_insert_metric_name': False,
        }
    else:
        some_args_about_checkpoint = {
            'save_top_k': 1,
            'filename': 'best'
        }

    checkpoint_callback = ModelCheckpoint(
        monitor=opt['monitor_metric'],
        mode=opt['monitor_mode'],
        save_last=True,
        dirpath=opt["checkpoint_path"],
        save_weights_only=True,
        **some_args_about_checkpoint
    )
    logger = TensorBoardLogger(opt["checkpoint_path"])

    # by defining callbacks below, The trainer will automatically log the learning rate and save models
    callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback,]
    
    extra_args = {}

    image_preprocess_func = None
    if hasattr(model.captioner, 'backbone') and model.captioner.backbone is not None:
        image_preprocess_func = model.captioner.backbone.get_preprocess_func('i')

    train_loader = get_loader(opt, 'train', print_info=False, image_preprocess_func=image_preprocess_func)
    vali_loader = get_loader(opt, 'validate', print_info=False, image_preprocess_func=image_preprocess_func)
    test_loader = get_loader(opt, 'test', print_info=False, image_preprocess_func=image_preprocess_func)

    opt['max_steps'] = len(train_loader) * opt['epochs']
    print('maximun training steps: {} * {} = {}'.format(len(train_loader), opt['epochs'], opt['max_steps']))
    
    trainer = Trainer(
        deterministic=True,
        weights_summary='full',
        auto_lr_find=False, 
        log_every_n_steps=50,
        max_epochs=opt['epochs'],
        max_steps=opt['max_steps'],
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=None,
        check_val_every_n_epoch=opt['check_val_every_n_epoch'],
        callbacks=callbacks,
        logger=logger,
        gpus=opt['gpus'],
        gradient_clip_val=opt['gradient_clip_val'],
        num_sanity_val_steps=0,
        **extra_args
    )

    trainer.fit(model, train_loader, vali_loader)

    print('best_model_path:', checkpoint_callback.best_model_path)
    print('best_model_score', checkpoint_callback.best_model_score)

    model = Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, test_loader)
