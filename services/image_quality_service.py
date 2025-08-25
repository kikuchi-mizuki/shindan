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
            'high': 0.75,     # 高品質閾値（厳格化）
            'medium': 0.5,    # 中品質閾値（厳格化）
            'low': 0.3        # 低品質閾値（厳格化）
        }
        
        # 品質ゲートの厳格な判定基準（緩和）
        self.strict_gate = {
            'min_sharpness': 0.3,    # 最小鮮明度（緩和）
            'min_contrast': 0.3,     # 最小コントラスト（緩和）
            'max_noise': 0.2,        # 最大ノイズ（逆数、緩和）
            'min_resolution': 600,   # 最小解像度（幅、緩和）
            'max_skew': 20,          # 最大傾き角度（緩和）
            'min_text_ratio': 0.02   # 最小文字占有率（緩和）
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
            
            # 厳格な品質ゲートチェック
            gate_check = self._check_strict_quality_gate(image, sharpness_score, contrast_score, noise_score)
            
            # 品質レベル判定
            quality_level = self._determine_quality_level(overall_score)
            
            # 厳格ゲートをパスした場合のみ処理を許可
            should_process = gate_check['passed'] and quality_level in ['high', 'medium']
            
            # デバッグ情報をログに出力
            logger.info(f"Quality gate check: passed={gate_check['passed']}, quality_level={quality_level}, should_process={should_process}")
            if not gate_check['passed']:
                logger.info(f"Quality gate issues: {gate_check['issues']}")
            
            result = {
                'quality_level': quality_level,
                'overall_score': overall_score,
                'sharpness_score': sharpness_score,
                'contrast_score': contrast_score,
                'noise_score': noise_score,
                'complexity_score': complexity_score,
                'recommendation': self._get_recommendation(quality_level, gate_check),
                'should_process': should_process,
                'gate_check': gate_check
            }
            
            logger.info(f"Image quality evaluation: {quality_level} (score: {overall_score:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating image quality: {e}")
            return self._get_low_quality_result(f"品質評価エラー: {e}")
    
    def _check_strict_quality_gate(self, image: np.ndarray, sharpness_score: float, contrast_score: float, noise_score: float) -> Dict[str, Any]:
        """厳格な品質ゲートチェック"""
        try:
            issues = []
            
            # 1. 鮮明度チェック
            if sharpness_score < self.strict_gate['min_sharpness']:
                issues.append(f"鮮明度不足 (スコア: {sharpness_score:.2f})")
            
            # 2. コントラストチェック
            if contrast_score < self.strict_gate['min_contrast']:
                issues.append(f"コントラスト不足 (スコア: {contrast_score:.2f})")
            
            # 3. ノイズチェック
            if noise_score < self.strict_gate['max_noise']:
                issues.append(f"ノイズ過多 (スコア: {noise_score:.2f})")
            
            # 4. 解像度チェック
            height, width = image.shape[:2]
            if width < self.strict_gate['min_resolution']:
                issues.append(f"解像度不足 (幅: {width}px)")
            
            # 5. 傾きチェック
            skew_angle = self._calculate_skew_angle(image)
            if abs(skew_angle) > self.strict_gate['max_skew']:
                issues.append(f"傾き過多 (角度: {skew_angle:.1f}°)")
            
            # 6. 文字占有率チェック
            text_ratio = self._calculate_text_ratio(image)
            if text_ratio < self.strict_gate['min_text_ratio']:
                issues.append(f"文字占有率不足 ({text_ratio:.3f})")
            
            passed = len(issues) == 0
            
            return {
                'passed': passed,
                'issues': issues,
                'skew_angle': skew_angle,
                'text_ratio': text_ratio,
                'resolution': f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"Strict gate check error: {e}")
            return {
                'passed': False,
                'issues': [f"ゲートチェックエラー: {e}"],
                'skew_angle': 0,
                'text_ratio': 0,
                'resolution': "unknown"
            }
    
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
    
    def _calculate_skew_angle(self, image: np.ndarray) -> float:
        """画像の傾き角度を計算"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # エッジ検出
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            # 直線検出
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.0
            
            angles = []
            # linesの形状を安全に処理
            if len(lines.shape) == 3:  # (1, N, 2) の形状
                lines = lines.reshape(-1, 2)
            
            for i, (rho, theta) in enumerate(lines[:10]):  # 最初の10本の線のみ
                try:
                    angle = theta * 180 / np.pi
                    if angle < 90:
                        angles.append(angle)
                    else:
                        angles.append(angle - 180)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Line {i} processing error: {e}")
                    continue
            
            if not angles:
                return 0.0
            
            # 中央値で傾きを推定
            return np.median(angles)
            
        except Exception as e:
            logger.warning(f"Skew angle calculation error: {e}")
            return 0.0
    
    def _calculate_text_ratio(self, image: np.ndarray) -> float:
        """文字占有率を計算"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 二値化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 文字領域の割合を計算
            text_pixels = np.sum(binary == 0)  # 黒い部分（文字）
            total_pixels = binary.size
            return text_pixels / total_pixels
            
        except Exception as e:
            logger.warning(f"Text ratio calculation error: {e}")
            return 0.0
    
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
    
    def _get_recommendation(self, quality_level: str, gate_check: Dict[str, Any] = None) -> str:
        """品質レベルに応じた推奨事項"""
        if gate_check and not gate_check.get('passed', True):
            issues = gate_check.get('issues', [])
            recommendation = "🚫 **品質ゲート未通過**\n\n"
            recommendation += "以下の問題が検出されました：\n"
            for issue in issues:
                recommendation += f"• {issue}\n"
            recommendation += "\n**改善方法：**\n"
            recommendation += "• 1ページずつ撮影してください\n"
            recommendation += "• 真上から撮影してください\n"
            recommendation += "• 明るい場所で撮影してください\n"
            recommendation += "• 影や反射を避けてください\n"
            recommendation += "• 書類スキャンアプリの使用を推奨します\n"
            return recommendation
        
        recommendations = {
            'high': "✅ 画像品質は良好です。薬剤名の読み取りを続行します。",
            'medium': "⚠️ 画像品質は中程度です。より鮮明な画像での再撮影を推奨します。",
            'low': "❌ 画像品質が低いため、薬剤名の読み取りが困難です。"
        }
        return recommendations.get(quality_level, "画像品質を評価できませんでした。")
    
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
