import logging
import traceback
import sys
from datetime import datetime

# 로거 설정
def setup_logger(log_file='training_errors.log'):
    """에러 로깅을 위한 로거 설정"""
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러
    print(f"log file is saved at {log_file}")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# 사용 예시
def train_with_error_logging():
    epoch = 40
    logger = setup_logger()
    
    try:
        # 여기에 실제 학습 코드
        from doclayout_yolo import YOLOv10
        # ... 학습 코드 ...
        asdf
    except Exception as e:
        error_msg = f"Training failed at epoch {epoch}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # 추가적으로 시스템 정보도 저장
        import psutil
        logger.error(f"Memory usage: {psutil.virtual_memory().percent}%")
        logger.error(f"Available memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
        
        raise  # 에러를 다시 raise하여 프로그램 종료

if __name__ == "__main__":
    train_with_error_logging()
