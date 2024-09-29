from PIL import Image
import torch

class DummyImageProcessor:
    def preprocess(self, images, return_tensors='pt'):
        return {'pixel_values': torch.rand(len(images), 3, 224, 224)}  # 임의로 (Batch, Channels, Height, Width) 크기의 텐서 반환

class DummyVisionTower:
    def __init__(self):
        self.image_processor = DummyImageProcessor()

    def __call__(self, processed_image):
        return torch.rand(1, 1024)  
    
class DummyImageProcessor2:
    def preprocess(self, images, return_tensors='pt'):
        return {'pixel_values': torch.rand(len(images), 3, 448, 448)}  # 임의로 (Batch, Channels, Height, Width) 크기의 텐서 반환
    
class DummyVisionTower2:
    def __init__(self):
        self.image_processor = DummyImageProcessor()

    def __call__(self, processed_image):
        return torch.rand(1, 1156)  


vision_towers = [DummyVisionTower(), DummyVisionTower2()]

# expand2square 함수
def expand2square(pil_imgs, background_color):
    arr_img = []
    for pil_img in pil_imgs:
        width, height = pil_img.size
        print(width, height)
        if width == height:
            arr_img.append(pil_img)
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            arr_img.append(result)
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            arr_img.append(result)
    print(arr_img)
    return arr_img

# 이미지 파일 경로 예시
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
pil_imgs = [Image.open(image_path) for image_path in image_paths]

# 비전 타워를 통해 피처 추출
features = []
for vision_tower in vision_towers:
    try:
        # 회색 배경으로 이미지를 정사각형으로 확장
        squared_x = expand2square(pil_imgs, tuple(int(t * 255) for t in [0.5, 0.5, 0.5]))

        # 이미지 전처리
        processed_image = vision_tower.image_processor.preprocess(squared_x, return_tensors='pt')['pixel_values']

        # 피처 추출
        feature = vision_tower(processed_image)
    except Exception as e:
        # 만약 preprocess 메서드가 없는 경우 직접 전처리 처리
        print(f"Error occurred: {e}. Falling back to direct processing.")
        processed_image = vision_tower.image_processor(pil_imgs, return_tensors='pt')
        feature = vision_tower(**processed_image)

    # 추출된 피처 저장
    features.append(feature)

# 결과 피처 출력 (예시로 출력만, 실제로는 다른 방식으로 사용할 수 있음)
for i, feature in enumerate(features):
    print(f"Feature from vision_tower {i}: {feature.shape}")
