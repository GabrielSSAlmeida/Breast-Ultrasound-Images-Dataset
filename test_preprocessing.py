#!/usr/bin/env python3
"""
Test script for the preprocessing pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_pipeline import UltrasoundPreprocessor, demonstrate_preprocessing

def test_preprocessing():
    """Test the preprocessing pipeline with a sample image"""
    
    print("🧪 Testando o Pipeline de Pré-processamento")
    print("=" * 50)
    
    # Test image path
    test_image_path = 'Dataset_BUSI_with_GT/benign/benign (1).png'
    
    # Load image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"❌ Erro: Não foi possível carregar a imagem {test_image_path}")
        return
    
    print(f"✅ Imagem carregada: {image.shape}")
    
    # Create preprocessor
    preprocessor = UltrasoundPreprocessor(target_size=(224, 224))
    
    # Test individual functions
    print("\n🔧 Testando funções individuais:")
    
    # Grayscale conversion
    gray = preprocessor.to_grayscale(image)
    print(f"   ✅ Conversão para grayscale: {gray.shape}")
    
    # Noise reduction
    denoised = preprocessor.gaussian_noise_reduction(gray)
    denoised = preprocessor.median_noise_reduction(denoised)
    print(f"   ✅ Redução de ruído: {denoised.shape}")
    
    # Contrast enhancement
    enhanced = preprocessor.contrast_enhancement(denoised)
    print(f"   ✅ Melhoria de contraste: {enhanced.shape}")
    
    # Background removal
    binary = preprocessor.background_removal(enhanced)
    print(f"   ✅ Remoção de fundo: {binary.shape}")
    
    # Edge detection
    edges = preprocessor.edge_detection(enhanced, method='canny')
    print(f"   ✅ Detecção de bordas: {edges.shape}")
    
    # Test complete pipelines
    print("\n🎯 Testando pipelines completos:")
    
    # Classification pipeline
    class_result = preprocessor.preprocess_for_classification(
        image,
        apply_grayscale=True,
        apply_noise_reduction=True,
        apply_contrast=True,
        apply_background_removal=False,
        apply_edge_detection=False
    )
    print(f"   ✅ Pipeline para classificação: {class_result.shape}")
    
    # Segmentation pipeline
    seg_result = preprocessor.preprocess_for_segmentation(
        image,
        apply_grayscale=True,
        apply_noise_reduction=True,
        apply_contrast=True,
        apply_structure_enhancement=True
    )
    print(f"   ✅ Pipeline para segmentação: {seg_result.shape}")
    
    # Binary mask creation
    binary_mask = preprocessor.create_binary_mask(image)
    print(f"   ✅ Máscara binária: {binary_mask.shape}")
    
    # Visualize results
    print("\n📊 Visualizando resultados...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Teste do Pipeline de Pré-processamento', fontsize=16)
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Enhanced
    axes[0, 2].imshow(enhanced, cmap='gray')
    axes[0, 2].set_title('Melhoria de Contraste')
    axes[0, 2].axis('off')
    
    # Binary
    axes[0, 3].imshow(binary, cmap='gray')
    axes[0, 3].set_title('Remoção de Fundo')
    axes[0, 3].axis('off')
    
    # Edges
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Detecção de Bordas')
    axes[1, 0].axis('off')
    
    # Classification result
    axes[1, 1].imshow(class_result, cmap='gray')
    axes[1, 1].set_title('Classificação')
    axes[1, 1].axis('off')
    
    # Segmentation result
    axes[1, 2].imshow(seg_result, cmap='gray')
    axes[1, 2].set_title('Segmentação')
    axes[1, 2].axis('off')
    
    # Binary mask
    axes[1, 3].imshow(binary_mask, cmap='gray')
    axes[1, 3].set_title('Máscara Binária')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Teste concluído com sucesso!")
    print("\n📋 Resumo:")
    print(f"   - Imagem original: {image.shape}")
    print(f"   - Pipeline para classificação: {class_result.shape}")
    print(f"   - Pipeline para segmentação: {seg_result.shape}")
    print(f"   - Máscara binária: {binary_mask.shape}")
    print("\n🎉 Pipeline funcionando corretamente!")

if __name__ == "__main__":
    test_preprocessing() 