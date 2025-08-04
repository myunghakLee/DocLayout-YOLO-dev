import argparse
from doclayout_yolo import YOLOv10
import logging
import traceback
import sys
from datetime import datetime
import error_loging

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, required=True, type=str)
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--epoch', default=None, required=True, type=int)
    parser.add_argument('--optimizer', default='auto', required=False, type=str)
    parser.add_argument('--momentum', default=0.9, required=False, type=float)
    parser.add_argument('--lr0', default=0.02, required=False, type=float)
    parser.add_argument('--warmup-epochs', default=3.0, required=False, type=float)
    parser.add_argument('--batch-size', default=16, required=False, type=int)
    parser.add_argument('--image-size', default=None, required=True, type=int)
    parser.add_argument('--mosaic', default=1.0, required=False, type=float)
    parser.add_argument('--rotate90', default=0.0, required=False, type=float)
    parser.add_argument('--pretrain', default=None, required=False, type=str)
    parser.add_argument('--val', default=1, required=False, type=int)
    parser.add_argument('--val-period', default=1, required=False, type=int)
    parser.add_argument('--plot', default=1, required=False, type=int)
    parser.add_argument('--project', default=None, required=True, type=str)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction)
    parser.add_argument('--workers', default=4, required=False, type=int)
    parser.add_argument('--device', default="0", required=False, type=str)
    parser.add_argument('--save-period', default=10, required=False, type=int)
    parser.add_argument('--patience', default=100, required=False, type=int)
    parser.add_argument('--name', default="yolo_checkpoints", required=False, type=str)
    args = parser.parse_args()
    
    # using '.pt' will load pretrained model
    if args.pretrain is not None:
        if args.pretrain == 'coco':
            model = f'yolov10{args.model}.pt'
            pretrain_name = 'coco'
        elif 'pt' in args.pretrain:
            model = args.pretrain
            if 'bestfit' in args.pretrain:
                pretrain_name = 'bestfit_layout'
            else:
                pretrain_name = "unknown"
    else:
        model = f'yolov10{args.model}.yaml'
        pretrain_name = 'None'

    
    
    # Load a pre-trained model
    if args.pretrain == "DocStructBench":
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
        
        filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-D4LA-from_scratch", filename="doclayout_yolo_d4la_imgsz1600_from_scratch.pt")
        model = YOLOv10(filepath)
        
        # One can optionally push this to the hub
        # model.push_to_hub("juliozhao/DocLayout-YOLO-D4LA-from_scratch")
        
        # Can now be reloaded as follows (and will increment download count)
        model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-D4LA-from_scratch")

        
    elif args.pretrain == "doclayout_yolo_docsynth300k_imgsz1600.pt":
        model = YOLOv10(args.pretrain)
    else:
        model = YOLOv10(model)
    
    # whether to val during training
    if args.val:
        val = True
    else:
        val = False
        
    # whether to plot
    if args.plot:
        plot = True
    else:
        plot = False
    
    # Train the model
    name = f"{args.name}/yolov10{args.model}_{args.data.split('/')[-2]}_epoch{args.epoch}_imgsz{args.image_size}_bs{args.batch_size}_rotate90_{args.rotate90}_pretrain_{pretrain_name}"

    
    try:
        results = model.train(
            data=f'{args.data}.yaml', 
            epochs=args.epoch, 
            warmup_epochs=args.warmup_epochs,
            lr0=args.lr0,
            optimizer=args.optimizer,
            momentum=args.momentum,
            imgsz=args.image_size, 
            mosaic=args.mosaic,
            rotate90=args.rotate90,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=plot,
            exist_ok=False,
            val=val,
            val_period=args.val_period,
            resume=args.resume,
            save_period=args.save_period,
            patience=args.patience,
            project=args.project, 
            name=name,
        )
    except Exception as e:

        import os
        save_path = f"{args.project}/{name}/training_errors.log"
        for i in range(2, 10000):
            print(f"check!!: {args.project}/{name}{i}")
            if os.path.isdir(f"{args.project}/{name}{i}"):
                save_path = f"{args.project}/{name}{i}/training_errors.log"
            else:
                # save_path = f"{args.project}/{name}{i}/training_errors.log"
                break
        logger = error_loging.setup_logger(save_path)
        print("="*1200)
    
        
        print(f"error message is saved at : {save_path}")    
        error_msg = f"Error Message: {str(e)}"
        
        logger.error(error_msg)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # 추가적으로 시스템 정보도 저장
        import psutil
        logger.error(f"Memory usage: {psutil.virtual_memory().percent}%")
        logger.error(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
        
        raise  # 에러를 다시 raise하여 프로그램 종료

        