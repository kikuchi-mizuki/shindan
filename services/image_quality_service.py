import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class ImageQualityService:
    """画像品質評価サービス"""
    
    def __init__(self):
        self.quality_thresholds = {
            'high': 0.7,      # 高品質閾値（緩和）
            'medium': 0.4,    # 中品質閾値（緩和）
            'low': 0.2        # 低品質閾値（緩和）
        }
    
    def evaluate_image_quality(self, image_path: str) -> Dict[str, Any]:
        """画像品質を評価"""
        try:
            # 画像読み込み
            image = cv2.imread(image_path)
            if image is None:
                return self._get_low_quality_result("画像の読み込みに失敗しました")
            
            # 品質評価指標を計算
            sharpness_score = self._calculate_sharpness(image)
            contrast_score = self._calculate_contrast(image)
            noise_score = self._calculate_noise_level(image)
            complexity_score = self._calculate_complexity(image)
            
            # 総合品質スコア
            overall_score = (sharpness_score + contrast_score + noise_score + complexity_score) / 4
            
            # 品質レベル判定
            quality_level = self._determine_quality_level(overall_score)
            
            result = {
                'quality_level': quality_level,
                'overall_score': overall_score,
                'sharpness_score': sharpness_score,
                'contrast_score': contrast_score,
                'noise_score': noise_score,
                'complexity_score': complexity_score,
                'recommendation': self._get_recommendation(quality_level),
                'should_process': quality_level in ['high', 'medium']
            }
            
            logger.info(f"Image quality evaluation: {quality_level} (score: {overall_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating image quality: {e}")
            return self._get_low_quality_result(f"品質評価エラー: {e}")
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """鮮明度を計算（ラプラシアン分散）"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # 正規化（0-1の範囲に）- 閾値を緩和
            sharpness_score = min(laplacian_var / 300, 1.0)
            # 最小値を設定して、比較的クリアな画像でも適切なスコアを付与
            return max(sharpness_score, 0.3)
        except:
            return 0.5
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """コントラストを計算"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 標準偏差でコントラストを評価
            contrast = np.std(gray)
            # 正規化（0-1の範囲に）- 閾値を緩和
            contrast_score = min(contrast / 30, 1.0)
            # 最小値を設定
            return max(contrast_score, 0.3)
        except:
            return 0.5
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """ノイズレベルを計算"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # ガウシアンフィルタでノイズを評価
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_level = np.mean(noise)
            # ノイズが少ないほど高スコア
            noise_score = max(0, 1.0 - (noise_level / 20))
            return noise_score
        except:
            return 0.5
    
    def _calculate_complexity(self, image: np.ndarray) -> float:
        """画像の複雑度を計算"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # エッジ検出で複雑度を評価
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            # 複雑度評価を改善 - メモアプリのような比較的単純な画像も適切に評価
            if edge_density < 0.005:  # 非常に単純（白紙など）
                complexity_score = 0.2
            elif edge_density < 0.02:  # 適度に単純（メモアプリなど）
                complexity_score = 0.7
            elif edge_density > 0.15:  # 非常に複雑
                complexity_score = 0.4
            else:  # 適度な複雑度
                complexity_score = 0.8
            return complexity_score
        except:
            return 0.5
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """品質レベルを判定"""
        if overall_score >= self.quality_thresholds['high']:
            return 'high'
        elif overall_score >= self.quality_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _get_recommendation(self, quality_level: str) -> str:
        """品質レベルに応じた推奨事項"""
        recommendations = {
            'high': "高品質な画像です。通常の処理を実行します。",
            'medium': "中品質な画像です。処理を実行しますが、結果を確認してください。",
            'low': "低品質な画像です。手動入力または画像の改善をお勧めします。"
        }
        return recommendations.get(quality_level, "品質評価に失敗しました")
    
    def _get_low_quality_result(self, error_message: str) -> Dict[str, Any]:
        """低品質結果を返す"""
        return {
            'quality_level': 'low',
            'overall_score': 0.0,
            'sharpness_score': 0.0,
            'contrast_score': 0.0,
            'noise_score': 0.0,
            'complexity_score': 0.0,
            'recommendation': error_message,
            'should_process': False
        }
    
    def generate_quality_guide(self, quality_level: str) -> str:
        """品質レベルに応じたガイドを生成"""
        guides = {
            'high': self._generate_high_quality_guide(),
            'medium': self._generate_medium_quality_guide(),
            'low': self._generate_low_quality_guide()
        }
        return guides.get(quality_level, "ガイドの生成に失敗しました")
    
    def _generate_high_quality_guide(self) -> str:
        """高品質画像用ガイド"""
        return """✅ **高品質な画像です！**
        
薬剤の検出と診断を実行します。
結果を確認して「診断」ボタンを押してください。"""
    
    def _generate_medium_quality_guide(self) -> str:
        """中品質画像用ガイド"""
        return """⚠️ **中品質な画像です**

薬剤の検出を試みますが、以下の点にご注意ください：

• 検出結果を必ず確認してください
• 不足している薬剤があれば手動で追加してください
• 誤検出があれば修正してください

**手動追加方法**: 「薬剤追加：薬剤名」と入力"""
    
    def _generate_low_quality_guide(self) -> str:
        """低品質画像用ガイド"""
        return """❌ **画像の品質が低いため、自動検出が困難です**

**推奨される対処法：**

1️⃣ **画像の改善**
• 明るい場所で撮影
• カメラを安定させる
• 文字がはっきり見えるようにする
• 影や反射を避ける

2️⃣ **手動入力**
以下の形式で薬剤を入力してください：
```
薬剤追加：クラリスロマイシン
薬剤追加：ベルソムラ
薬剤追加：デビゴ
```

3️⃣ **メモアプリの使用**
• 薬剤名をメモアプリに記入
• スクリーンショットを撮影
• 再度送信してください

**例**:
```
クラリスロマイシン
ベルソムラ
デビゴ
ロゼレム
フルボキサミン
アムロジピン
エソメプラゾール
```"""
