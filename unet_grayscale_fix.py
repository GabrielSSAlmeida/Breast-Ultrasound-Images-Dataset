"""
Soluções para o erro de incompatibilidade de canais no U-Net
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# SOLUÇÃO 1: Converter grayscale para RGB (mantém o modelo original)
def load_images_and_masks_rgb(base_path, classes=('benign', 'malignant'), img_size=(256, 256)):
    """
    Carrega imagens e máscaras convertendo grayscale para RGB
    """
    img_list, mask_list, label_list = [], [], []
    
    from preprocessing_pipeline import UltrasoundPreprocessor
    preprocessor = UltrasoundPreprocessor(target_size=img_size)

    for class_name in classes:
        folder = os.path.join(base_path, class_name)
        print(f"🔄 Carregando imagens e máscaras para {class_name} (RGB)...")
        
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

            # Pré-processamento
            processed_img = preprocessor.preprocess_for_segmentation(
                img,
                apply_grayscale=True,
                apply_noise_reduction=True,
                apply_contrast=True,
                apply_structure_enhancement=True
            )
            
            # Converte grayscale para 3 canais (RGB)
            if len(processed_img.shape) == 2:
                processed_img = np.stack([processed_img] * 3, axis=-1)
            
            # Redimensiona
            processed_img = cv2.resize(processed_img, img_size)
            
            # Processa máscara
            mask = cv2.resize(mask, img_size).astype('float32') / 255.0
            mask = np.expand_dims(mask, axis=-1)

            img_list.append(processed_img)
            mask_list.append(mask)
            label_list.append(class_name)
        
        print(f"   ✅ {len([x for x in label_list if x == class_name])} imagens processadas")

    return np.array(img_list), np.array(mask_list), np.array(label_list)

# SOLUÇÃO 2: Modificar U-Net para aceitar grayscale (recomendado para imagens médicas)
def build_unet_grayscale(input_shape=(256, 256, 1)):
    """
    U-Net modificado para aceitar imagens grayscale (1 canal)
    """
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D()(c3)
    u1 = Concatenate()([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = UpSampling2D()(c4)
    u2 = Concatenate()([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

def load_images_and_masks_grayscale(base_path, classes=('benign', 'malignant'), img_size=(256, 256)):
    """
    Carrega imagens e máscaras mantendo grayscale (1 canal)
    """
    img_list, mask_list, label_list = [], [], []
    
    from preprocessing_pipeline import UltrasoundPreprocessor
    preprocessor = UltrasoundPreprocessor(target_size=img_size)

    for class_name in classes:
        folder = os.path.join(base_path, class_name)
        print(f"🔄 Carregando imagens e máscaras para {class_name} (Grayscale)...")
        
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

            # Pré-processamento
            processed_img = preprocessor.preprocess_for_segmentation(
                img,
                apply_grayscale=True,
                apply_noise_reduction=True,
                apply_contrast=True,
                apply_structure_enhancement=True
            )
            
            # Mantém grayscale (1 canal)
            if len(processed_img.shape) == 2:
                processed_img = np.expand_dims(processed_img, axis=-1)
            
            # Redimensiona
            processed_img = cv2.resize(processed_img, img_size)
            
            # Processa máscara
            mask = cv2.resize(mask, img_size).astype('float32') / 255.0
            mask = np.expand_dims(mask, axis=-1)

            img_list.append(processed_img)
            mask_list.append(mask)
            label_list.append(class_name)
        
        print(f"   ✅ {len([x for x in label_list if x == class_name])} imagens processadas")

    return np.array(img_list), np.array(mask_list), np.array(label_list)

# Exemplo de uso das duas soluções:
"""
# SOLUÇÃO 1: Converter para RGB (mantém modelo original)
X_seg, y_seg, labels = load_images_and_masks_rgb('Dataset_BUSI_with_GT', img_size=(256, 256))
print(f"Shape das imagens (RGB): {X_seg.shape}")  # (N, 256, 256, 3)

# Usar modelo original
model_seg = build_model()  # input_shape=(256, 256, 3)

# SOLUÇÃO 2: Modificar modelo para grayscale (recomendado)
X_seg, y_seg, labels = load_images_and_masks_grayscale('Dataset_BUSI_with_GT', img_size=(256, 256))
print(f"Shape das imagens (Grayscale): {X_seg.shape}")  # (N, 256, 256, 1)

# Usar modelo modificado
model_seg = build_unet_grayscale()  # input_shape=(256, 256, 1)
model_seg.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
""" 