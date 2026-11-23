from sahi.annotation import BoundingBox
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import check_requirements
from sahi.postprocess.combine import PostprocessPredictions
from sahi.postprocess.utils import ObjectPredictionList
from typing import List, Dict, Optional, Tuple
from shapely.geometry import Polygon
import numpy as np
from shapely.validation import make_valid
class AdvancedPositionDrivenCombiner:
    """
    位置驱动的高级合并器 (完全复现 BatchPatchMerger 逻辑)

    核心改进：
    1. 边缘惩罚机制 (Edge Penalty)
    2. **IoF包含关系过滤 (在NMS之前执行)** ← 关键修正
    3. 标准NMS
    """

    def __init__(
            self,
            # 边缘惩罚参数
            edge_penalty_ratio: float = 0.6,
            edge_margin: int = 32,

            # IoF 过滤参数
            iof_threshold: float = 0.85,
            size_ratio_threshold: float = 0.6,

            # NMS 参数
            match_metric: str = "IOU",
            match_threshold: float = 0.5,

            # 调试开关
            verbose: bool = False,
    ):
        """
        Args:
            edge_penalty_ratio: 边缘框置信度惩罚系数 (0-1)
            edge_margin: 边缘区域宽度 (像素)
            iof_threshold: IoF阈值，超过则认为被包含 (0-1)
            size_ratio_threshold: 面积比阈值，小于则认为是碎片 (0-1)
            match_metric: NMS匹配度量 ("IOU" or "IOS")
            match_threshold: NMS匹配阈值
            verbose: 是否打印逐步调试信息（与 val_dual_2 对比验证时使用）
        """
        self.edge_penalty_ratio = max(0.0, min(1.0, edge_penalty_ratio))
        self.edge_margin = max(0, edge_margin)
        self.iof_threshold = iof_threshold
        self.size_ratio_threshold = size_ratio_threshold
        self.match_metric = match_metric
        self.match_threshold = match_threshold
        self.verbose = verbose

        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'invalid_filtered': 0,
            'edge_penalized': 0,
            'iof_filtered': 0,
            'nms_removed': 0,
            'final_count': 0
        }

    # ==================== 几何工具函数 ====================

    @staticmethod
    def _get_polygon(prediction: ObjectPrediction) -> Optional[Polygon]:
        """将预测框转换为 Shapely Polygon

        修改说明：
        原来的实现只用 bbox.to_voc_bbox() 得到外接矩形来构建 Polygon，
        对于 OBB（旋转框）来说精度损失严重，IoF 和 NMS 计算都会偏差。

        修改后：优先从 mask.segmentation[0] 读取真实 OBB 的 8 个坐标点
        （x1,y1,x2,y2,x3,y3,x4,y4），这是 ultralytics.py 在 is_obb 分支
        存入的原始旋转框坐标，与 val_dual_2_chengfa_iof.py 的处理方式一致。
        只有在没有 segmentation 时才回退到外接矩形。
        """
        try:
            # ✅ 修改：优先使用真实 OBB 坐标（存储在 mask.segmentation[0] 中）
            # OBB 格式：[x1, y1, x2, y2, x3, y3, x4, y4]（已映射到全图坐标）
            if prediction.mask is not None and prediction.mask.segmentation:
                seg = prediction.mask.segmentation[0]  # [x1,y1,x2,y2,x3,y3,x4,y4]
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)  # shape (4, 2)
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = make_valid(poly)
                return poly

            # ⬇ 原来的实现（回退：无 OBB 时用外接矩形）
            # 原代码问题：对旋转框只取外接矩形，IoF/NMS 计算面积偏大，
            # 导致本不重叠的旋转框被误判为重叠，或包含关系判断不准确。
            # bbox = prediction.bbox.to_voc_bbox()  # [x_min, y_min, x_max, y_max]
            # x_min, y_min, x_max, y_max = bbox
            # poly = Polygon([
            #     (x_min, y_min),
            #     (x_max, y_min),
            #     (x_max, y_max),
            #     (x_min, y_max)
            # ])
            bbox = prediction.bbox.to_voc_bbox()
            x_min, y_min, x_max, y_max = bbox
            poly = Polygon([
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ])
            if not poly.is_valid:
                poly = make_valid(poly)
            return poly
        except Exception as e:
            return None

    @staticmethod
    def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
        """计算 IoU (交集/并集)"""
        try:
            if not poly1 or not poly2 or not poly1.is_valid or not poly2.is_valid:
                return 0.0
            inter = poly1.intersection(poly2).area
            union = poly1.union(poly2).area
            return inter / union if union > 0 else 0.0
        except:
            return 0.0

    @staticmethod
    def compute_iof(poly_small: Polygon, poly_large: Polygon) -> float:
        """
        计算 IoF (Intersection over Foreground)
        Returns: intersection / area(poly_small)
        """
        try:
            if not poly_small or not poly_large:
                return 0.0
            small_area = poly_small.area
            if small_area <= 0:
                return 0.0
            inter = poly_small.intersection(poly_large).area
            return inter / small_area
        except:
            return 0.0

    def apply_edge_penalty(self, prediction, slice_bboxes, full_shape, slice_shape=None):
        """
        边缘惩罚：与 val_dual_2_chengfa_iof.py._apply_edge_penalty 完全一致。

        当 prediction.slice_origin 存在时（方案A），精确使用该预测所属 slice 的
        边界判断，避免对其他 slice 边界的误判。
        当 slice_origin 不存在时回退到近似模式（遍历所有 slice_bboxes）。
        """
        img_h, img_w = full_shape

        # 优先使用真实 OBB 坐标
        if prediction.mask is not None and prediction.mask.segmentation:
            seg = prediction.mask.segmentation[0]
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
            x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
            x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        else:
            x_min, y_min, x_max, y_max = prediction.bbox.to_voc_bbox()

        is_edge = False

        if hasattr(prediction, 'slice_origin') and slice_shape is not None:
            # ✅ 精确模式：只用该预测所属的 slice 边界（与 val_dual_2 完全一致）
            slice_h, slice_w = slice_shape
            pos_x, pos_y = prediction.slice_origin[0], prediction.slice_origin[1]
            patch_left = pos_x
            patch_right = pos_x + slice_w
            patch_top = pos_y
            patch_bottom = pos_y + slice_h

            if (x_min < patch_left + self.edge_margin) and (patch_left > 0):      is_edge = True
            if (x_max > patch_right - self.edge_margin) and (patch_right < img_w):  is_edge = True
            if (y_min < patch_top + self.edge_margin) and (patch_top > 0):      is_edge = True
            if (y_max > patch_bottom - self.edge_margin) and (patch_bottom < img_h):  is_edge = True
        else:
            # 回退模式：遍历所有 slice_bboxes（近似，可能多惩罚）
            for s_bbox in slice_bboxes:
                sx1, sy1, sx2, sy2 = s_bbox
                if abs(x_min - sx1) < self.edge_margin and sx1 > 0: is_edge = True
                if abs(x_max - sx2) < self.edge_margin and sx2 < img_w: is_edge = True
                if abs(y_min - sy1) < self.edge_margin and sy1 > 0: is_edge = True
                if abs(y_max - sy2) < self.edge_margin and sy2 < img_h: is_edge = True
                if is_edge: break

        if is_edge:
            old_score = prediction.score.value
            prediction.score.value *= self.edge_penalty_ratio
            self.stats['edge_penalized'] += 1
            if self.verbose:
                # 找出是哪条边界触发了惩罚
                reasons = []
                if hasattr(prediction, 'slice_origin') and slice_shape is not None:
                    slice_h, slice_w = slice_shape
                    pos_x2, pos_y2 = prediction.slice_origin
                    if (x_min < pos_x2 + self.edge_margin) and (pos_x2 > 0):          reasons.append("left")
                    if (x_max > pos_x2 + slice_w - self.edge_margin) and (pos_x2 + slice_w < img_w): reasons.append(
                        "right")
                    if (y_min < pos_y2 + self.edge_margin) and (pos_y2 > 0):          reasons.append("top")
                    if (y_max > pos_y2 + slice_h - self.edge_margin) and (pos_y2 + slice_h < img_h): reasons.append(
                        "bottom")
                    print(f"    [EdgePenalty] slice=({pos_x2},{pos_y2}) "
                          f"box=({x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}) "
                          f"score: {old_score:.4f} → {prediction.score.value:.4f}  触发边: {reasons}")
        return prediction

    # def filter_enclosures(self, prediction_list: ObjectPredictionList) -> ObjectPredictionList:
    #     if len(prediction_list) <= 1:
    #         return prediction_list

    #     preds = prediction_list.tolist()
    #     # 1. 必须按面积从大到小排序，确保是“大框”过滤“小框”
    #     polys = [self._get_polygon(p) for p in preds]
    #     areas = [poly.area if poly else 0 for poly in polys]

    #     # 获取按面积降序排列的索引
    #     sorted_idx = np.argsort(areas)[::-1]
    #     keep_indices = set(range(len(preds)))
    #     removed_indices = set()

    #     for i_idx in range(len(sorted_idx)):
    #         i = sorted_idx[i_idx]
    #         if i in removed_indices: continue

    #         for j_idx in range(i_idx + 1, len(sorted_idx)):
    #             j = sorted_idx[j_idx]
    #             if j in removed_indices: continue

    #             # 计算小框 j 在大框 i 中的比例 (IoF)
    #             iof = self.compute_iof(polys[j], polys[i])

    #             # 【改进】：如果小框被高度包含，直接移除，无需严格的面积比限制
    #             if iof > self.iof_threshold:
    #                 # 额外检查：除非小框置信度远高于大框，否则移除小框
    #                 if preds[j].score.value < preds[i].score.value * 1.5:
    #                     removed_indices.add(j)
    #                     keep_indices.discard(j)
    #                     self.stats['iof_filtered'] += 1

    #     return ObjectPredictionList([preds[i] for i in sorted(keep_indices)])
    def filter_enclosures(self, prediction_list: ObjectPredictionList) -> ObjectPredictionList:
        """
        ✅ 核心修正：完全复现 BatchPatchMerger 的包含关系过滤逻辑
        """
        if len(prediction_list) <= 1:
            return prediction_list

        preds = prediction_list.tolist()
        n = len(preds)

        # 1. 预计算所有 Polygon 和面积
        polys = [self._get_polygon(p) for p in preds]
        areas = [poly.area if poly and poly.is_valid else 0 for poly in polys]

        # 2. 初始化保留集合
        keep_indices = set(range(n))
        removed_indices = set()

        # 3. ✅ 关键修正：直接使用原始索引，无需排序
        # BatchPatchMerger 的逻辑是：O(N^2) 暴力比较
        for i in range(n):
            if i in removed_indices or polys[i] is None:
                continue

            for j in range(n):
                if i == j or j in removed_indices or polys[j] is None:
                    continue

                # ✅ 核心逻辑：判断 i 是否是 j 的子集碎片

                # (1) 面积预筛选：如果 i >= j * threshold，则 i 不可能是 j 的碎片
                if areas[i] >= areas[j] * self.size_ratio_threshold:
                    continue

                # (2) 计算 i 在 j 中的比例 (IoF)
                iof = self.compute_iof(polys[i], polys[j])

                # (3) 判定删除条件
                if iof > self.iof_threshold:
                    removed_indices.add(i)
                    keep_indices.discard(i)
                    self.stats['iof_filtered'] += 1
                    if self.verbose:
                        si = preds[i].mask.segmentation[0] if (preds[i].mask and preds[i].mask.segmentation) else None
                        sj = preds[j].mask.segmentation[0] if (preds[j].mask and preds[j].mask.segmentation) else None
                        pi_center = (np.array(si).reshape(-1, 2).mean(axis=0) if si else None)
                        pj_center = (np.array(sj).reshape(-1, 2).mean(axis=0) if sj else None)
                        print(f"    [IoF_Remove] 小框#{i} (area={areas[i]:.0f}, score={preds[i].score.value:.4f}, "
                              f"center={pi_center}) 被大框#{j} (area={areas[j]:.0f}, score={preds[j].score.value:.4f}, "
                              f"center={pj_center}) 吸收  IoF={iof:.3f}")
                    break  # i 已经被删除，跳出内层循环

        # 4. 返回保留的预测
        return ObjectPredictionList([preds[i] for i in sorted(keep_indices)])

    def apply_nms(
            self,
            prediction_list: ObjectPredictionList
    ) -> ObjectPredictionList:
        """
        应用标准 NMS
        """
        if len(prediction_list) <= 1:
            return prediction_list

        predictions = prediction_list.tolist()
        polys = [self._get_polygon(pred) for pred in predictions]

        # 按置信度降序排列
        sorted_indices = sorted(
            range(len(predictions)),
            key=lambda i: predictions[i].score.value,
            reverse=True
        )

        keep_indices = []
        removed = set()

        for i in sorted_indices:
            if i in removed or not polys[i]:
                continue

            keep_indices.append(i)

            for j in sorted_indices:
                if j == i or j in removed or not polys[j]:
                    continue

                # 计算匹配度
                if self.match_metric == "IOU":
                    match = self.compute_iou(polys[i], polys[j])
                else:  # IOS
                    match = self.compute_iof(polys[j], polys[i])

                if match > self.match_threshold:
                    removed.add(j)
                    self.stats['nms_removed'] += 1
                    if self.verbose:
                        si = predictions[i].mask.segmentation[0] if (
                                    predictions[i].mask and predictions[i].mask.segmentation) else None
                        sj = predictions[j].mask.segmentation[0] if (
                                    predictions[j].mask and predictions[j].mask.segmentation) else None
                        ci = np.array(si).reshape(-1, 2).mean(axis=0) if si else None
                        cj = np.array(sj).reshape(-1, 2).mean(axis=0) if sj else None
                        print(f"    [NMS_Remove]  保留#{i}(score={predictions[i].score.value:.4f}, center={ci}) "
                              f"抑制#{j}(score={predictions[j].score.value:.4f}, center={cj})  IOU={match:.3f}")

        kept_predictions = [predictions[i] for i in keep_indices]
        return ObjectPredictionList(kept_predictions)

    # ==================== 主接口 ====================

    def __call__(
            self,
            object_prediction_list: List[ObjectPrediction],
            slice_bboxes: List[List[int]],
            full_shape: Tuple[int, int],
            slice_shape: Tuple[int, int]
    ) -> List[ObjectPrediction]:
        """
        主处理流程 (完全复现 BatchPatchMerger 逻辑)

        Args:
            object_prediction_list: 所有切片的预测列表
            slice_bboxes: 每个切片的边界框列表
            full_shape: 完整图像尺寸 (H, W)
            slice_shape: 切片尺寸 (H, W)

        Returns:
            处理后的预测列表
        """
        # 重置统计
        self.stats = {k: 0 for k in self.stats.keys()}
        self.stats['total_predictions'] = len(object_prediction_list)

        if not object_prediction_list:
            if self.verbose:
                print("  [Combiner] 输入为空，跳过所有处理")
            return []

        img_h, img_w = full_shape

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  [Combiner] 开始后处理 | 图像尺寸: {img_w}x{img_h}")
            print(f"  [Combiner] 切片尺寸: {slice_shape} | 切片数: {len(slice_bboxes)}")
            print(f"  [Combiner] 输入预测框总数: {len(object_prediction_list)}")
            scores = [p.score.value for p in object_prediction_list]
            print(f"  [Combiner] 原始置信度 min={min(scores):.4f} max={max(scores):.4f} "
                  f"avg={sum(scores) / len(scores):.4f}")
            print(f"{'=' * 60}")

        # ✅ Step 0: 有效性过滤 + 坐标裁剪（完全复现 val_dual_2._is_box_valid + _clip_box_to_image）
        valid_predictions = []
        for pred in object_prediction_list:
            # ????????? valid ????? val_dual_2 ????? clip
            if pred.mask is not None and pred.mask.segmentation:
                seg = pred.mask.segmentation[0]
                pts_raw = np.array(seg, dtype=np.float32).reshape(-1, 2)
            else:
                x1, y1, x2, y2 = pred.bbox.to_voc_bbox()
                pts_raw = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            x_min, y_min = pts_raw[:, 0].min(), pts_raw[:, 1].min()
            x_max, y_max = pts_raw[:, 0].max(), pts_raw[:, 1].max()
            box_w = x_max - x_min
            box_h = y_max - y_min

            # 1. ?????? min_size=5?
            if box_w < 5 or box_h < 5:
                self.stats['invalid_filtered'] = self.stats.get('invalid_filtered', 0) + 1
                continue

            # 2. ?????????
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
            if cx < 0 or cx >= img_w or cy < 0 or cy >= img_h:
                self.stats['invalid_filtered'] = self.stats.get('invalid_filtered', 0) + 1
                continue

            # 3. ????????? 10% ???
            margin_w, margin_h = img_w * 0.1, img_h * 0.1
            if x_min < -margin_w or x_max > img_w + margin_w:
                self.stats['invalid_filtered'] = self.stats.get('invalid_filtered', 0) + 1
                continue
            if y_min < -margin_h or y_max > img_h + margin_h:
                self.stats['invalid_filtered'] = self.stats.get('invalid_filtered', 0) + 1
                continue

            pts = pts_raw.copy()
            pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)

            if pred.mask is not None and pred.mask.segmentation:
                pred.mask.segmentation[0] = pts.flatten().tolist()
            else:
                pred.bbox = BoundingBox([
                    float(pts[:, 0].min()),
                    float(pts[:, 1].min()),
                    float(pts[:, 0].max()),
                    float(pts[:, 1].max()),
                ])

            valid_predictions.append(pred)

        object_prediction_list = valid_predictions

        if self.verbose:
            print(f"\n  [Step 0] 有效性过滤: {self.stats['total_predictions']} → "
                  f"{len(object_prediction_list)} (移除 {self.stats['invalid_filtered']} 个无效框)")

        # ✅ Step 1: 应用边缘惩罚
        if self.verbose:
            print(f"\n  [Step 1] 边缘惩罚 (ratio={self.edge_penalty_ratio}, margin={self.edge_margin}px):")
        for pred in object_prediction_list:
            self.apply_edge_penalty(pred, slice_bboxes, full_shape, slice_shape)
        if self.verbose:
            print(f"  [Step 1] 共惩罚 {self.stats['edge_penalized']} 个边缘框")

        pred_list = ObjectPredictionList(object_prediction_list)

        # ✅ Step 2: IoF 包含关系过滤 (关键：在NMS之前！)
        if self.verbose:
            print(f"\n  [Step 2] IoF包含过滤 (iof_thr={self.iof_threshold}, size_ratio={self.size_ratio_threshold}):")
        n_before_iof = len(pred_list)
        pred_list = self.filter_enclosures(pred_list)
        if self.verbose:
            print(f"  [Step 2] IoF过滤: {n_before_iof} → {len(pred_list)} "
                  f"(移除 {self.stats['iof_filtered']} 个被包含小框)")

        # ✅ Step 3: NMS
        if self.verbose:
            print(f"\n  [Step 3] NMS (metric={self.match_metric}, thr={self.match_threshold}):")
        n_before_nms = len(pred_list)
        pred_list = self.apply_nms(pred_list)
        if self.verbose:
            print(f"  [Step 3] NMS: {n_before_nms} → {len(pred_list)} "
                  f"(抑制 {self.stats['nms_removed']} 个重叠框)")

        self.stats['final_count'] = len(pred_list)

        if self.verbose:
            print(f"\n  [Combiner] 最终结果: {self.stats['final_count']} 个框")
            if len(pred_list) > 0:
                final_preds = pred_list.tolist() if isinstance(pred_list.tolist(), list) else [pred_list.tolist()]
                scores_final = [p.score.value for p in
                                (final_preds if isinstance(final_preds, list) else [final_preds])]
                if scores_final:
                    print(f"  [Combiner] 最终置信度 min={min(scores_final):.4f} "
                          f"max={max(scores_final):.4f} avg={sum(scores_final) / len(scores_final):.4f}")
            print(f"{'=' * 60}\n")

        # 强制返回 list，避免 ObjectPredictionList.tolist() 在长度为1时返回单个对象
        result = pred_list.tolist()
        if not isinstance(result, list):
            result = [result]
        return result

    def get_stats(self) -> Dict:
        """获取处理统计信息"""
        return self.stats.copy()
class AdvancedPostprocessPredictions(PostprocessPredictions):
    """
    扩展 SAHI 原生 PostprocessPredictions 类
    集成高级位置驱动算法
    """

    def __init__(
            self,
            match_threshold: float = 0.5,
            match_metric: str = "IOU",
            # 新增参数
            use_advanced_combiner: bool = True,
            edge_penalty_ratio: float = 0.6,
            edge_margin: int = 32,
            iof_threshold: float = 0.85,
            size_ratio_threshold: float = 0.6,
            verbose: bool = False,
    ):
        """
        Args:
            match_threshold: NMS 匹配阈值
            match_metric: 匹配度量 ("IOU" or "IOS")
            use_advanced_combiner: 是否使用高级合并器
            edge_penalty_ratio: 边缘惩罚系数
            edge_margin: 边缘区域宽度
            iof_threshold: IoF 阈值
            size_ratio_threshold: 面积比阈值
            verbose: 是否打印逐步调试信息
        """
        super().__init__(
            match_threshold=match_threshold,
            match_metric=match_metric
        )

        self.use_advanced_combiner = use_advanced_combiner

        if use_advanced_combiner:
            self.advanced_combiner = AdvancedPositionDrivenCombiner(
                edge_penalty_ratio=edge_penalty_ratio,
                edge_margin=edge_margin,
                iof_threshold=iof_threshold,
                size_ratio_threshold=size_ratio_threshold,
                match_metric=match_metric,
                match_threshold=match_threshold,
                verbose=verbose,
            )

    def __call__(
            self,
            object_prediction_list: List[ObjectPrediction],
            # 新增参数
            slice_bboxes: Optional[List[List[int]]] = None,
            full_shape: Optional[Tuple[int, int]] = None,
            slice_shape: Optional[Tuple[int, int]] = None,
    ) -> ObjectPredictionList:
        """
        执行后处理

        Args:
            object_prediction_list: 预测列表
            slice_bboxes: 切片边界框列表 (for advanced mode)
            full_shape: 完整图像尺寸 (for advanced mode)
            slice_shape: 切片尺寸 (for advanced mode)
        """
        if self.use_advanced_combiner and slice_bboxes and full_shape and slice_shape:
            # 使用高级合并器
            processed = self.advanced_combiner(
                object_prediction_list,
                slice_bboxes,
                full_shape,
                slice_shape
            )
            return ObjectPredictionList(processed)
        else:
            # 回退到原生 SAHI 处理
            return super().__call__(object_prediction_list)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if self.use_advanced_combiner:
            return self.advanced_combiner.get_stats()
        return {}