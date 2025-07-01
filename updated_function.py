import os
import cv2
import numpy as np
from preprocessing_pipeline import UltrasoundPreprocessor

def load_images_and_masks(base_path, classes=('benign', 'malignant'), img_size=(256, 256)):
    """
    Carrega imagens e máscaras com pré-processamento aplicado
    
    Args:
        base_path: Caminho base do dataset
        classes: Tupla com as classes a serem carregadas
        img_size: Tamanho das imagens (width, height)
    
    Returns:
        img_list: Array de imagens pré-processadas (3 canais)
        mask_list: Array de máscaras
        label_list: Array de labels
    """
    img_list, mask_list, label_list = [], [], []
    
    # Inicializa o pré-processador
    preprocessor = UltrasoundPreprocessor(target_size=img_size)

    for class_name in classes:
        folder = os.path.join(base_path, class_name)
        print(f"🔄 Carregando imagens e máscaras para {class_name} com pré-processamento...")
        
        for fname in os.listdir(folder):
            if 'mask' in fname.lower():
                continue
            img_path = os.path.join(folder, fname)
            mask_path = os.path.join(folder, fname.split('.')[0] + '_mask.png')
            if not os.path.exists(mask_path): 
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            # Aplica pré-processamento otimizado para segmentação
            processed_img = preprocessor.preprocess_for_segmentation(
                img,
                apply_grayscale=True,
                apply_noise_reduction=True,
                apply_contrast=True,
                apply_structure_enhancement=True
            )
            
            # Converte grayscale para 3 canais (RGB) para compatibilidade com U-Net
            if len(processed_img.shape) == 2:
                processed_img = np.stack([processed_img] * 3, axis=-1)
            
            # Redimensiona para o tamanho desejado
            processed_img = cv2.resize(processed_img, img_size)
            
            # Processa máscara
            mask = cv2.resize(mask, img_size).astype('float32') / 255.0
            mask = np.expand_dims(mask, axis=-1)

            img_list.append(processed_img)
            mask_list.append(mask)
            label_list.append(class_name)
        
        print(f"   ✅ {len([x for x in label_list if x == class_name])} imagens processadas para {class_name}")

    return np.array(img_list), np.array(mask_list), np.array(label_list)

# Exemplo de uso:
# X_seg, y_seg, labels = load_images_and_masks('Dataset_BUSI_with_GT', classes=('benign', 'malignant'), img_size=(256, 256))
# print(f"Dataset carregado: {len(X_seg)} imagens")
# print(f"Shape das imagens: {X_seg.shape}")
# print(f"Shape das máscaras: {y_seg.shape}")
# print(f"Classes: {np.unique(labels)}") 