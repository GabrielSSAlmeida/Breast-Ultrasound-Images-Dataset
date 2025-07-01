"""
Pipeline completo de pré-processamento para imagens de ultrassom da mama
Inclui: conversão para grayscale, redução de ruído, melhoria de contraste,
segmentação, remoção de fundo e realce de estruturas
"""

import cv2
import numpy as np
from skimage import filters, morphology, measure
from scipy import ndimage
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
import matplotlib.pyplot as plt

class UltrasoundPreprocessor:
    """
    Pipeline completo de pré-processamento para imagens de ultrassom da mama
    Inclui: conversão para grayscale, redução de ruído, melhoria de contraste,
    segmentação, remoção de fundo e realce de estruturas
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def to_grayscale(self, image):
        """Conversão para grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def gaussian_noise_reduction(self, image, kernel_size=5, sigma=1.0):
        """Redução de ruído com filtro Gaussiano"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def median_noise_reduction(self, image, kernel_size=5):
        """Redução de ruído com filtro mediano"""
        return cv2.medianBlur(image, kernel_size)
    
    def contrast_enhancement(self, image, method='clahe'):
        """Melhoria de contraste"""
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)
        elif method == 'histogram_equalization':
            return cv2.equalizeHist(image)
        elif method == 'gamma_correction':
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        return image
    
    def background_removal(self, image, method='otsu'):
        """Remoção de fundo e binarização"""
        if method == 'otsu':
            # Binarização de Otsu
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            # Binarização adaptativa
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        elif method == 'watershed':
            # Segmentação por watershed
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Operações morfológicas para melhorar a segmentação
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        else:
            # Threshold simples
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def edge_detection(self, image, method='canny'):
        """Detecção de bordas"""
        if method == 'canny':
            # Detecção de bordas Canny
            edges = canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
            return edges.astype(np.uint8) * 255
        elif method == 'sobel':
            # Operadores de Sobel
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.uint8(sobel * 255 / sobel.max())
            return sobel
        elif method == 'laplacian':
            # Operador Laplaciano
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            return laplacian
        return image
    
    def structure_enhancement(self, image, method='morphological'):
        """Realce de estruturas"""
        if method == 'morphological':
            # Operações morfológicas para realçar estruturas
            kernel = np.ones((3,3), np.uint8)
            # Abertura para remover ruído
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            # Fechamento para preencher gaps
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            return closed
        elif method == 'tophat':
            # Top-hat transform para realçar estruturas claras
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            return tophat
        return image
    
    def remove_small_objects(self, image, min_size=100):
        """Remove objetos pequenos (ruído)"""
        # Remove componentes conectados pequenos
        labeled_image = measure.label(image)
        props = measure.regionprops(labeled_image)
        
        # Cria máscara para objetos grandes
        mask = np.zeros_like(image, dtype=bool)
        for prop in props:
            if prop.area >= min_size:
                mask[labeled_image == prop.label] = True
        
        return mask.astype(np.uint8) * 255
    
    def preprocess_for_classification(self, image, apply_grayscale=True, 
                                    apply_noise_reduction=True, apply_contrast=True,
                                    apply_background_removal=False, apply_edge_detection=False):
        """
        Pré-processamento otimizado para classificação
        """
        processed = image.copy()
        
        if apply_grayscale:
            processed = self.to_grayscale(processed)
        
        if apply_noise_reduction:
            # Aplica filtros de redução de ruído
            processed = self.gaussian_noise_reduction(processed, kernel_size=5, sigma=1.0)
            processed = self.median_noise_reduction(processed, kernel_size=3)
        
        if apply_contrast:
            # Melhoria de contraste
            processed = self.contrast_enhancement(processed, method='clahe')
        
        if apply_background_removal:
            # Remoção de fundo (opcional para classificação)
            binary = self.background_removal(processed, method='otsu')
            # Usa a máscara binária para mascarar a imagem original
            processed = cv2.bitwise_and(processed, processed, mask=binary)
        
        if apply_edge_detection:
            # Detecção de bordas (opcional para classificação)
            edges = self.edge_detection(processed, method='canny')
            # Combina imagem original com bordas
            processed = cv2.addWeighted(processed, 0.7, edges, 0.3, 0)
        
        # Redimensionamento
        processed = cv2.resize(processed, self.target_size)
        
        # Normalização
        processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def preprocess_for_segmentation(self, image, apply_grayscale=True,
                                  apply_noise_reduction=True, apply_contrast=True,
                                  apply_structure_enhancement=True):
        """
        Pré-processamento otimizado para segmentação
        """
        processed = image.copy()
        
        if apply_grayscale:
            processed = self.to_grayscale(processed)
        
        if apply_noise_reduction:
            # Redução de ruído mais suave para segmentação
            processed = self.gaussian_noise_reduction(processed, kernel_size=3, sigma=0.8)
        
        if apply_contrast:
            # Melhoria de contraste
            processed = self.contrast_enhancement(processed, method='clahe')
        
        if apply_structure_enhancement:
            # Realce de estruturas
            processed = self.structure_enhancement(processed, method='morphological')
        
        # Redimensionamento
        processed = cv2.resize(processed, self.target_size)
        
        # Normalização
        processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def create_binary_mask(self, image, method='otsu'):
        """
        Cria máscara binária para segmentação
        """
        processed = self.to_grayscale(image)
        processed = self.gaussian_noise_reduction(processed, kernel_size=3, sigma=0.8)
        processed = self.contrast_enhancement(processed, method='clahe')
        
        # Binarização
        binary = self.background_removal(processed, method=method)
        
        # Remove objetos pequenos
        binary = self.remove_small_objects(binary, min_size=50)
        
        # Redimensionamento
        binary = cv2.resize(binary, self.target_size)
        
        # Normalização
        binary = binary.astype(np.float32) / 255.0
        
        return binary
    
    def visualize_preprocessing_steps(self, image, title="Pipeline de Pré-processamento"):
        """
        Visualiza todas as etapas do pré-processamento
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(title, fontsize=16)
        
        # Imagem original
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Grayscale
        gray = self.to_grayscale(image)
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # Redução de ruído
        denoised = self.gaussian_noise_reduction(gray)
        denoised = self.median_noise_reduction(denoised)
        axes[0, 2].imshow(denoised, cmap='gray')
        axes[0, 2].set_title('Redução de Ruído')
        axes[0, 2].axis('off')
        
        # Melhoria de contraste
        enhanced = self.contrast_enhancement(denoised)
        axes[0, 3].imshow(enhanced, cmap='gray')
        axes[0, 3].set_title('Melhoria de Contraste')
        axes[0, 3].axis('off')
        
        # Remoção de fundo
        binary = self.background_removal(enhanced)
        axes[1, 0].imshow(binary, cmap='gray')
        axes[1, 0].set_title('Remoção de Fundo')
        axes[1, 0].axis('off')
        
        # Detecção de bordas
        edges = self.edge_detection(enhanced, method='canny')
        axes[1, 1].imshow(edges, cmap='gray')
        axes[1, 1].set_title('Detecção de Bordas (Canny)')
        axes[1, 1].axis('off')
        
        # Realce de estruturas
        enhanced_struct = self.structure_enhancement(enhanced)
        axes[1, 2].imshow(enhanced_struct, cmap='gray')
        axes[1, 2].set_title('Realce de Estruturas')
        axes[1, 2].axis('off')
        
        # Resultado final para classificação
        final_class = self.preprocess_for_classification(image)
        axes[1, 3].imshow(final_class, cmap='gray')
        axes[1, 3].set_title('Resultado Final (Classificação)')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()

# Função auxiliar para pré-processar lote de imagens
def preprocess_batch(images, task='classification', **kwargs):
    """
    Pré-processa um lote de imagens
    
    Args:
        images: Lista ou array de imagens
        task: 'classification' ou 'segmentation'
        **kwargs: Parâmetros específicos do pré-processamento
    """
    preprocessor = UltrasoundPreprocessor(target_size=(224, 224))
    processed_images = []
    
    for image in images:
        if task == 'classification':
            processed = preprocessor.preprocess_for_classification(image, **kwargs)
        else:  # segmentation
            processed = preprocessor.preprocess_for_segmentation(image, **kwargs)
        processed_images.append(processed)
    
    return np.array(processed_images)

# Função para demonstrar o pipeline
def demonstrate_preprocessing(image_path):
    """
    Demonstra o pipeline de pré-processamento em uma imagem
    """
    preprocessor = UltrasoundPreprocessor()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return
    
    print("🔧 Demonstração do Pipeline de Pré-processamento")
    print("=" * 50)
    
    # Visualiza todas as etapas
    preprocessor.visualize_preprocessing_steps(image)
    
    # Mostra resultados específicos
    print("\n📊 Resultados do Pré-processamento:")
    
    # Para classificação
    class_result = preprocessor.preprocess_for_classification(
        image, 
        apply_grayscale=True,
        apply_noise_reduction=True,
        apply_contrast=True,
        apply_background_removal=False,
        apply_edge_detection=False
    )
    
    # Para segmentação
    seg_result = preprocessor.preprocess_for_segmentation(
        image,
        apply_grayscale=True,
        apply_noise_reduction=True,
        apply_contrast=True,
        apply_structure_enhancement=True
    )
    
    # Máscara binária
    binary_mask = preprocessor.create_binary_mask(image)
    
    # Visualização dos resultados finais
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(class_result, cmap='gray')
    axes[0].set_title('Pré-processamento para Classificação')
    axes[0].axis('off')
    
    axes[1].imshow(seg_result, cmap='gray')
    axes[1].set_title('Pré-processamento para Segmentação')
    axes[1].axis('off')
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Máscara Binária')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Demonstração concluída!")

if __name__ == "__main__":
    print("🔧 Pipeline de pré-processamento para imagens de ultrassom da mama")
    print("📋 Funcionalidades disponíveis:")
    print("   - Conversão para grayscale")
    print("   - Redução de ruído (Gaussiano + Mediano)")
    print("   - Melhoria de contraste (CLAHE, Histograma, Gamma)")
    print("   - Remoção de fundo e binarização")
    print("   - Detecção de bordas (Canny, Sobel, Laplaciano)")
    print("   - Realce de estruturas (Morphological, Top-hat)")
    print("   - Pré-processamento otimizado para classificação e segmentação") 