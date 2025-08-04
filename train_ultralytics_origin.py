import argparse
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
import albumentations as A

if __name__ == "__main__":


    from ultralytics import YOLO
    
    # Load a COCO-pretrained YOLO11n model
    
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
    parser.add_argument('--pretrain', default=None, required=False, type=str)
    parser.add_argument('--val', default=1, required=False, type=int)
    parser.add_argument('--val-period', default=1, required=False, type=int)
    parser.add_argument('--plot', default=0, required=False, type=int)
    parser.add_argument('--project', default=None, required=True, type=str)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction)
    parser.add_argument('--workers', default=4, required=False, type=int)
    parser.add_argument('--device', default="0", required=False, type=str)
    parser.add_argument('--save-period', default=10, required=False, type=int)
    parser.add_argument('--patience', default=100, required=False, type=int)
    parser.add_argument('--rotate90', default=0.0, required=False, type=float)
    args = parser.parse_args()
    
    model = YOLO(f"{args.model}.pt")
    alb_aug = Albumentations(
        A.Compose(
            [A.RandomRotate90(p=args.rotate90)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
        )
    )
    # Train the model
    name = f"{args.model}_{args.data.split('/')[-2]}_imgsz{args.image_size}_pretrain_coco_rotate90_{args.rotate90}"
    results = model.train(
        data=f'{args.data}.yaml', 
        epochs=args.epoch, 
        warmup_epochs=args.warmup_epochs,
        lr0=args.lr0,
        optimizer=args.optimizer,
        momentum=args.momentum,
        imgsz=args.image_size, 
        mosaic=args.mosaic,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        val=True,
        resume=args.resume,
        save_period=args.save_period,
        patience=args.patience,
        project=args.project, 
        name=name,
        augment=alb_aug
    )