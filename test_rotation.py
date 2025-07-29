#!/usr/bin/env python3
"""
90도 회전 augmentation 테스트 스크립트
"""

import numpy as np
import cv2
from doclayout_yolo.data.augment import Random90Rotation
from doclayout_yolo.utils.instance import Instances

def test_90_rotation():
    """90도 회전 augmentation 테스트"""
    
    # 테스트 이미지 생성 (100x100 픽셀)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 테스트 bounding boxes 생성 (xywh 형식)
    # center_x, center_y, width, height (normalized)
    bboxes = np.array([
        [0.5, 0.5, 0.3, 0.2],  # 중앙 박스
        [0.2, 0.3, 0.1, 0.15], # 좌상단 박스
    ])
    
    # Instances 객체 생성
    instances = Instances(bboxes=bboxes, bbox_format="xywh", normalized=True)
    
    # 라벨 딕셔너리 생성
    labels = {
        "img": img,
        "instances": instances,
        "cls": np.array([0, 1])  # 클래스 라벨
    }
    
    # Random90Rotation 객체 생성 (확률 1.0으로 설정하여 항상 회전)
    rotate_transform = Random90Rotation(p=1.0)
    
    print("Original image shape:", img.shape)
    print("Original bboxes:", bboxes)
    
    # 여러 번 테스트
    for i in range(5):
        test_labels = {
            "img": labels["img"].copy(),
            "instances": Instances(bboxes=labels["instances"].bboxes.copy(), 
                                  bbox_format="xywh", 
                                  normalized=True),
            "cls": labels["cls"].copy()
        }
        
        # 회전 적용
        result = rotate_transform(test_labels)
        
        print(f"\nTest {i+1}:")
        print(f"  Result image shape: {result['img'].shape}")
        print(f"  Result bboxes: {result['instances'].bboxes}")
        
    print("\n90도 회전 augmentation 테스트 완료!")

if __name__ == "__main__":
    test_90_rotation()
