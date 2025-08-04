# #!/bin/bash
# echo "▶ DEBUG: \$1 = <$1>"

# # 1) 인자 개수 확인
# if [ $# -lt 1 ]; then
#   echo "Usage: $0 <1|2|3|4>"
#   exit 1
# fi

# if ! [[ "$1" =~ ^[1-4]$ ]]; then
#   echo "Invalid option: $1"
#   echo "Usage: $0 <1|2|3|4>"
#   exit 1
# fi


# # 2) 첫 번째 인자($1)로 분기
# case "$1" in
#   1)
#     echo "1 명령 실행"
#     # python3 train_ultralytics_origin.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model yolo11m --epoch 1300 --image-size 1100 --pretrain coco --project ./yolov11m_rotate90 --batch-size 8 --workers 0 --rotate90 0.1
#     python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 16 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0

    
#     ;;
#   2)
#     echo "2 명령 실행"
#     # python3 train_ultralytics_origin.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model yolo11n --epoch 1300 --image-size 1100 --pretrain coco --project ./yolov11n_rotate90 --batch-size 8 --workers 0 --rotate90 0.1
#     python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 64 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.1 --workers 0
#     ;;
#   3)
#     echo "3 명령 실행"
#     # python3 train_ultralytics_origin.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model yolo11m --epoch 1300 --image-size 1100 --pretrain coco --project ./yolov11m_rotate90 --batch-size 8 --workers 0 --rotate90 0.2

#       python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 64 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.05 --workers 0
#     ;;

#   4)
#     echo "4 명령 실행"
#     # python3 train_ultralytics_origin.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model yolo11n --epoch 1300 --image-size 1100 --pretrain coco --project ./yolov11n_rotate90 --batch-size 8 --workers 0 --rotate90 0.2
#       python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 64 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.15 --workers 0
#     ;;
#   *)
#     echo "Invalid option: $1"
#     echo "Usage: $0 <1|2>"
#     exit 1
#     ;;
# esac


echo "▶ DEBUG: \$1 = <$1>"
case "$1" in
  1)
    echo "1 명령 실행"

    ## rotate
    python train.py --name pretrain_doc_rotate --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m-doclayout --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.00 --patience 200
    
    python train.py --name pretrain_doc_rotate --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m-doclayout --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.05 --patience 200
    
    python train.py --name pretrain_doc_rotate --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m-doclayout --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.10 --patience 200
    
    python train.py --name pretrain_doc_rotate --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m-doclayout --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.15 --patience 200
    
    python train.py --name pretrain_doc_rotate --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m-doclayout --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.20 --patience 200
    ;;

  2)
    echo "2 명령 실행"

    ## COCO Pretrain model size m
    python train.py --name pretrain_coco_rotate_size_m --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.00 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_m --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.05 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_m --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.10 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_m --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.15 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_m --data /data/myung/DocLayout-YOLO-dev/my_data/data --model m --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.20 --patience 200
    
    
    
    
    ## COCO Pretrain model size n
    python train.py --name pretrain_coco_rotate_size_n --data /data/myung/DocLayout-YOLO-dev/my_data/data --model n --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.00 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_n --data /data/myung/DocLayout-YOLO-dev/my_data/data --model n --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.05 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_n --data /data/myung/DocLayout-YOLO-dev/my_data/data --model n --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.10 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_n --data /data/myung/DocLayout-YOLO-dev/my_data/data --model n --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.15 --patience 200
    
    python train.py --name pretrain_coco_rotate_size_n --data /data/myung/DocLayout-YOLO-dev/my_data/data --model n --epoch 1200 --image-size 1600 --batch-size 24 --project yolov10_finetune_my_data_low_lr --plot 1 --optimizer SGD --lr0 0.01 --pretrain coco --workers 0 --rotate90 0.20 --patience 200
    ;;
esac



# No Pretrain

# python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 24 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.1 --workers 0

# python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 24 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.15 --workers 0


# python train.py --data /data/myung/DocLayout-YOLO-dev/vaiv_data/vaiv_dataset/data --model m-doclayout --epoch 600 --image-size 1600 --batch-size 24 --project yolov10_finetune_docsynth --plot 1 --optimizer SGD --lr0 0.04 --pretrain doclayout_yolo_docsynth300k_imgsz1600.pt --workers 0 --rotate90 0.15 --workers 0
