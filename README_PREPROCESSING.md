# 🔧 Pipeline de Pré-processamento para Imagens de Ultrassom da Mama

Este pipeline oferece técnicas avançadas de pré-processamento especificamente otimizadas para imagens de ultrassom da mama, incluindo classificação e segmentação.

## 📋 Funcionalidades Implementadas

### 1. **Conversão para Grayscale**
- Conversão automática de imagens coloridas para escala de cinza
- Preserva informações importantes para análise médica

### 2. **Redução de Ruído**
- **Filtro Gaussiano**: Suaviza a imagem removendo ruído gaussiano
- **Filtro Mediano**: Remove ruído impulsivo (salt & pepper)
- Parâmetros configuráveis para diferentes tipos de ruído

### 3. **Melhoria de Contraste**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Melhoria adaptativa de contraste
- **Equalização de Histograma**: Melhoria global de contraste
- **Correção Gamma**: Ajuste não-linear de brilho

### 4. **Segmentação e Remoção de Fundo**
- **Binarização de Otsu**: Threshold automático
- **Binarização Adaptativa**: Threshold local
- **Segmentação Watershed**: Para casos complexos
- Remoção de objetos pequenos (ruído)

### 5. **Detecção de Bordas**
- **Canny**: Detecção robusta de bordas
- **Sobel**: Operadores de gradiente
- **Laplaciano**: Detecção de bordas de segunda ordem

### 6. **Realce de Estruturas**
- **Operações Morfológicas**: Abertura e fechamento
- **Top-hat Transform**: Realce de estruturas claras
- Preservação de características anatômicas importantes

## 🚀 Como Usar

### Instalação de Dependências

```python
%pip install scikit-image scipy
```

### Importação do Pipeline

```python
from preprocessing_pipeline import UltrasoundPreprocessor, preprocess_batch, demonstrate_preprocessing

# Instância global do pré-processador
preprocessor = UltrasoundPreprocessor(target_size=(224, 224))
```

### Pré-processamento para Classificação

```python
# Carrega uma imagem
img = cv2.imread('imagem.png')

# Pré-processamento básico para classificação
processed_img = preprocessor.preprocess_for_classification(
    img,
    apply_grayscale=True,
    apply_noise_reduction=True,
    apply_contrast=True,
    apply_background_removal=False,  # Opcional
    apply_edge_detection=False       # Opcional
)
```

### Pré-processamento para Segmentação

```python
# Pré-processamento otimizado para segmentação
processed_img = preprocessor.preprocess_for_segmentation(
    img,
    apply_grayscale=True,
    apply_noise_reduction=True,
    apply_contrast=True,
    apply_structure_enhancement=True
)
```

### Criação de Máscaras Binárias

```python
# Cria máscara binária para segmentação
binary_mask = preprocessor.create_binary_mask(
    img, 
    method='otsu'  # 'otsu', 'adaptive', 'watershed'
)
```

### Pré-processamento em Lote

```python
# Lista de imagens
images = [img1, img2, img3, ...]

# Pré-processamento em lote para classificação
processed_batch = preprocess_batch(
    images, 
    task='classification',
    apply_grayscale=True,
    apply_noise_reduction=True,
    apply_contrast=True
)
```

## 📊 Visualização e Análise

### Demonstração Completa do Pipeline

```python
# Demonstra todas as etapas do pré-processamento
demonstrate_preprocessing('caminho/para/imagem.png')
```

### Visualização de Etapas Individuais

```python
# Visualiza todas as etapas do pipeline
preprocessor.visualize_preprocessing_steps(img, "Pipeline Completo")
```

### Comparação de Configurações

```python
def compare_configurations():
    configs = [
        {'name': 'Básico', 'params': {'apply_grayscale': True}},
        {'name': 'Com Ruído', 'params': {'apply_grayscale': True, 'apply_noise_reduction': True}},
        {'name': 'Completo', 'params': {'apply_grayscale': True, 'apply_noise_reduction': True, 'apply_contrast': True}}
    ]
    
    for config in configs:
        processed = preprocessor.preprocess_for_classification(img, **config['params'])
        # Visualizar resultado...
```

## 🔧 Configurações Avançadas

### Parâmetros de Redução de Ruído

```python
# Filtro Gaussiano personalizado
denoised = preprocessor.gaussian_noise_reduction(
    img, 
    kernel_size=7,  # Tamanho do kernel
    sigma=1.5       # Desvio padrão
)

# Filtro Mediano personalizado
denoised = preprocessor.median_noise_reduction(
    img, 
    kernel_size=5   # Tamanho do kernel
)
```

### Métodos de Melhoria de Contraste

```python
# CLAHE (recomendado)
enhanced = preprocessor.contrast_enhancement(img, method='clahe')

# Equalização de histograma
enhanced = preprocessor.contrast_enhancement(img, method='histogram_equalization')

# Correção gamma
enhanced = preprocessor.contrast_enhancement(img, method='gamma_correction')
```

### Detecção de Bordas

```python
# Canny (recomendado)
edges = preprocessor.edge_detection(img, method='canny')

# Sobel
edges = preprocessor.edge_detection(img, method='sobel')

# Laplaciano
edges = preprocessor.edge_detection(img, method='laplacian')
```

## 📈 Integração com Modelos Existentes

### Para VGG-16 (Classificação)

```python
# Modifique a célula de carregamento de dados
for idx, label in enumerate(classes):
    folder = os.path.join(base_path, label)
    for filename in os.listdir(folder):
        if 'mask' not in filename:
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            
            # Aplica pré-processamento
            processed_img = preprocessor.preprocess_for_classification(img)
            
            # Converte para 3 canais se necessário
            if len(processed_img.shape) == 2:
                processed_img = np.stack([processed_img] * 3, axis=-1)
            
            X.append(processed_img)
            y.append(idx)
```

### Para U-Net (Segmentação)

```python
def load_images_and_masks_with_preprocessing(class_folder):
    img_list = []
    mask_list = []
    folder = os.path.join(base_path, class_folder)
    
    for fname in os.listdir(folder):
        if 'mask' not in fname:
            img_path = os.path.join(folder, fname)
            mask_path = os.path.join(folder, fname.split('.')[0] + '_mask.png')
            
            if not os.path.exists(mask_path): 
                continue
                
            # Pré-processamento para segmentação
            img = cv2.imread(img_path)
            processed_img = preprocessor.preprocess_for_segmentation(img)
            
            # Processa máscara
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (128, 128))
            mask = np.expand_dims(mask, axis=-1) / 255.0
            
            img_list.append(processed_img)
            mask_list.append(mask)
    
    return img_list, mask_list
```

## 🎯 Recomendações por Tarefa

### Para Classificação
- ✅ Conversão para grayscale
- ✅ Redução de ruído (Gaussiano + Mediano)
- ✅ Melhoria de contraste (CLAHE)
- ⚠️ Remoção de fundo (opcional)
- ⚠️ Detecção de bordas (opcional)

### Para Segmentação
- ✅ Conversão para grayscale
- ✅ Redução de ruído (suave)
- ✅ Melhoria de contraste (CLAHE)
- ✅ Realce de estruturas
- ✅ Criação de máscaras binárias

## 🔍 Monitoramento de Qualidade

### Métricas de Qualidade

```python
def assess_preprocessing_quality(original, processed):
    # Contraste
    contrast_original = np.std(original)
    contrast_processed = np.std(processed)
    
    # SNR (Signal-to-Noise Ratio)
    snr_original = np.mean(original) / np.std(original)
    snr_processed = np.mean(processed) / np.std(processed)
    
    print(f"Contraste: {contrast_original:.2f} → {contrast_processed:.2f}")
    print(f"SNR: {snr_original:.2f} → {snr_processed:.2f}")
```

## 🚨 Considerações Importantes

1. **Preservação de Informações Médicas**: O pipeline foi projetado para preservar características anatômicas importantes
2. **Configuração de Parâmetros**: Ajuste os parâmetros conforme a qualidade das suas imagens
3. **Validação Médica**: Sempre valide os resultados com especialistas médicos
4. **Performance**: Para grandes datasets, considere processamento em paralelo

## 📞 Suporte

Para dúvidas ou sugestões sobre o pipeline de pré-processamento, consulte a documentação ou entre em contato.

---

**Nota**: Este pipeline foi desenvolvido especificamente para imagens de ultrassom da mama e pode precisar de ajustes para outros tipos de imagens médicas. 