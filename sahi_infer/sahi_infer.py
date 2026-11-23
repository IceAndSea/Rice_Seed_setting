#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAHI inference entry for 6-channel RGB+IR input without editing SAHI source."""

import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.engine.results import Masks

import sahi.predict
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.postprocess.combine import PostprocessPredictions
from sahi.prediction import PredictionResult
from sahi.predict import filter_predictions

from custom_combine import AdvancedPostprocessPredictions

DEBUG = False
SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_RATIO = 300 / 1024


def _to_prediction_result_image(image):
    if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] > 4:
        return np.ascontiguousarray(image[:, :, :3])
    return image


def _normalize_object_prediction_list(object_prediction_list):
    if object_prediction_list is None:
        return []
    if isinstance(object_prediction_list, list):
        return object_prediction_list
    if hasattr(object_prediction_list, "list"):
        return list(object_prediction_list.list)
    if hasattr(object_prediction_list, "tolist"):
        converted = object_prediction_list.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [object_prediction_list]


def _prediction_color_bgr(prediction):
    category_name = (prediction.category.name or "").strip().lower()
    if prediction.category.id == 0 or "full" in category_name:
        return (0, 0, 255)
    if prediction.category.id == 1 or "empty" in category_name:
        return (255, 0, 0)
    return (0, 255, 255)


def _draw_predictions_on_reflective_image(image_bgr, object_prediction_list):
    vis_image = image_bgr.copy()

    for prediction in _normalize_object_prediction_list(object_prediction_list):
        color = _prediction_color_bgr(prediction)

        has_mask = prediction.mask is not None and prediction.mask.segmentation
        is_obb_pred = has_mask and len(prediction.mask.segmentation) == 1 and len(prediction.mask.segmentation[0]) == 8

        if is_obb_pred:
            points = np.array(prediction.mask.segmentation[0], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_image, [points], isClosed=True, color=color, thickness=3)
        else:
            x1, y1, x2, y2 = map(int, prediction.bbox.to_xyxy())
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness=3)

    return vis_image


def _save_predictions_for_map(txt_path, object_prediction_list, image_width, image_height):
    with open(txt_path, "w", encoding="utf-8") as file:
        for prediction in _normalize_object_prediction_list(object_prediction_list):
            class_id = int(prediction.category.id)
            score = float(prediction.score.value)

            if prediction.mask is not None and prediction.mask.segmentation:
                points = np.array(prediction.mask.segmentation[0], dtype=np.float32).reshape(-1, 2)
            else:
                x1, y1, x2, y2 = prediction.bbox.to_xyxy()
                points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            points[:, 0] /= image_width
            points[:, 1] /= image_height
            coords = " ".join(f"{value:.6f}" for value in points.reshape(-1))
            file.write(f"{class_id} {coords} {score:.6f}\n")


def _get_val_dual2_slice_bboxes(image_height, image_width, slice_height, slice_width, overlap_height_ratio,
                                overlap_width_ratio):
    overlap_h = int(overlap_height_ratio * slice_height)
    overlap_w = int(overlap_width_ratio * slice_width)
    stride_h = max(1, slice_height - overlap_h)
    stride_w = max(1, slice_width - overlap_w)

    ys = list(range(0, max(1, image_height - slice_height + 1), stride_h))
    if ys[-1] != max(0, image_height - slice_height):
        ys.append(max(0, image_height - slice_height))

    xs = list(range(0, max(1, image_width - slice_width + 1), stride_w))
    if xs[-1] != max(0, image_width - slice_width):
        xs.append(max(0, image_width - slice_width))

    return [
        [x, y, min(x + slice_width, image_width), min(y + slice_height, image_height)]
        for y in ys
        for x in xs
        if x <= max(0, image_width - 1) and y <= max(0, image_height - 1)
    ]


class CustomUltralyticsModel(UltralyticsDetectionModel):
    """Ultralytics wrapper that accepts 6-channel RGB+IR numpy input."""

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 6:
            raise ValueError(f"Expected an HWC 6-channel numpy array, got shape {getattr(image, 'shape', None)}")

        image_rgb_style = image.copy()
        image_rgb_style[:, :, 0:3] = image[:, :, 0:3][:, :, ::-1]
        image_rgb_style[:, :, 3:6] = image[:, :, 3:6][:, :, ::-1]

        img_tensor = torch.from_numpy(image_rgb_style).permute(2, 0, 1).float().to(self.device)
        img_tensor /= 255.0

        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        orig_h, orig_w = image.shape[:2]
        new_h = ((orig_h - 1) // 32 + 1) * 32
        new_w = ((orig_w - 1) // 32 + 1) * 32

        if new_h != orig_h or new_w != orig_w:
            pad_h = new_h - orig_h
            pad_w = new_w - orig_w
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), "constant", 0)

        kwargs = {"verbose": False, "conf": self.confidence_threshold, "device": self.device}
        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size

        prediction_result = self.model.predict(img_tensor, **kwargs)

        for result in prediction_result:
            result.orig_shape = (orig_h, orig_w)

        final_predictions = []
        for result in prediction_result:
            if self.has_mask:
                if not result.masks:
                    device = getattr(self.model, "device", "cpu")
                    result.masks = Masks(torch.tensor([], device=device), (orig_h, orig_w))
                final_predictions.append((result.boxes.data, result.masks.data))
            elif self.is_obb:
                device = getattr(self.model, "device", "cpu")
                if result.obb is not None and len(result.obb) > 0:
                    obb_data = torch.cat(
                        [result.obb.xyxy, result.obb.conf.unsqueeze(-1), result.obb.cls.unsqueeze(-1)],
                        dim=1,
                    )
                    final_predictions.append((obb_data, result.obb.xyxyxyxy))
                else:
                    final_predictions.append(
                        (torch.empty((0, 6), device=device), torch.empty((0, 4, 2), device=device))
                    )
            else:
                final_predictions.append(result.boxes.data)

        self._original_predictions = final_predictions
        self._original_shape = image.shape


def _patched_get_prediction(
    image,
    detection_model,
    shift_amount=[0, 0],
    full_shape=None,
    postprocess=None,
    verbose=0,
    exclude_classes_by_name=None,
    exclude_classes_by_id=None,
):
    durations_in_seconds = {}
    if isinstance(image, np.ndarray):
        image_array = np.ascontiguousarray(image)
    else:
        image_array = np.ascontiguousarray(sahi.predict.read_image_as_pil(image))

    time_start = time.time()
    detection_model.perform_inference(image_array)
    durations_in_seconds["prediction"] = time.time() - time_start

    if full_shape is None:
        full_shape = [image_array.shape[0], image_array.shape[1]]

    time_start = time.time()
    detection_model.convert_original_predictions(shift_amount=shift_amount, full_shape=full_shape)
    object_prediction_list = filter_predictions(
        detection_model.object_prediction_list,
        exclude_classes_by_name,
        exclude_classes_by_id,
    )

    if postprocess is not None:
        object_prediction_list = postprocess(object_prediction_list)
    object_prediction_list = _normalize_object_prediction_list(object_prediction_list)

    durations_in_seconds["postprocess"] = time.time() - time_start
    if verbose == 1:
        print("Prediction performed in", durations_in_seconds["prediction"], "seconds.")

    return PredictionResult(
        image=_to_prediction_result_image(image),
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )


sahi.predict.get_prediction = _patched_get_prediction


def _resolve_postprocess(
    postprocess_class,
    postprocess_type,
    postprocess_match_threshold,
    postprocess_match_metric,
    postprocess_class_agnostic,
):
    if postprocess_class is None:
        postprocess_constructor = sahi.predict.POSTPROCESS_NAME_TO_CLASS[postprocess_type]
        return postprocess_constructor(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )

    if isinstance(postprocess_class, PostprocessPredictions):
        return postprocess_class

    if callable(postprocess_class):
        try:
            return postprocess_class(
                match_threshold=postprocess_match_threshold,
                match_metric=postprocess_match_metric,
                class_agnostic=postprocess_class_agnostic,
            )
        except TypeError:
            return postprocess_class()

    raise TypeError("postprocess_class should be a PostprocessPredictions instance or a callable.")


def _apply_postprocess(postprocess, object_prediction_list, slice_bboxes, full_shape, slice_shape):
    try:
        result = postprocess(
            object_prediction_list,
            slice_bboxes=slice_bboxes,
            full_shape=full_shape,
            slice_shape=slice_shape,
        )
    except TypeError:
        result = postprocess(object_prediction_list)
    return _normalize_object_prediction_list(result)


def _patched_get_sliced_prediction(
    image,
    detection_model=None,
    slice_height=None,
    slice_width=None,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    perform_standard_pred=True,
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",
    postprocess_match_threshold=0.5,
    postprocess_class_agnostic=False,
    verbose=1,
    merge_buffer_length=None,
    auto_slice_resolution=True,
    slice_export_prefix=None,
    slice_dir=None,
    exclude_classes_by_name=None,
    exclude_classes_by_id=None,
    postprocess_class=None,
):
    del slice_export_prefix, slice_dir

    if detection_model is None:
        raise ValueError("detection_model must be provided.")

    if not isinstance(image, np.ndarray):
        image = np.asarray(sahi.predict.read_image_as_pil(image))

    image = np.ascontiguousarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image data, got shape {image.shape}")

    image_h, image_w = image.shape[:2]
    durations_in_seconds = {}

    time_start = time.time()
    if auto_slice_resolution:
        raise ValueError("auto_slice_resolution=True is not supported in this val_dual_2-aligned slicer.")
    if slice_height is None or slice_width is None:
        raise ValueError("slice_height and slice_width must be provided.")

    slice_bboxes = _get_val_dual2_slice_bboxes(
        image_height=image_h,
        image_width=image_w,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    durations_in_seconds["slice"] = time.time() - time_start

    if detection_model.is_obb:
        postprocess_type = "NMS"

    postprocess = _resolve_postprocess(
        postprocess_class=postprocess_class,
        postprocess_type=postprocess_type,
        postprocess_match_threshold=postprocess_match_threshold,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_class_agnostic=postprocess_class_agnostic,
    )

    if verbose in (1, 2):
        print(f"Performing prediction on {len(slice_bboxes)} slices.")

    slice_shape = (
        slice_height if slice_height is not None else image_h,
        slice_width if slice_width is not None else image_w,
    )
    postprocess_time = 0.0
    object_prediction_list = []

    time_start = time.time()
    for x_min, y_min, x_max, y_max in slice_bboxes:
        slice_image = image[y_min:y_max, x_min:x_max]
        prediction_result = sahi.predict.get_prediction(
            image=slice_image,
            detection_model=detection_model,
            shift_amount=[x_min, y_min],
            full_shape=[image_h, image_w],
            exclude_classes_by_name=exclude_classes_by_name,
            exclude_classes_by_id=exclude_classes_by_id,
        )

        for object_prediction in prediction_result.object_prediction_list:
            if object_prediction:
                shifted_prediction = object_prediction.get_shifted_object_prediction()
                object.__setattr__(shifted_prediction, "slice_origin", [x_min, y_min])
                object_prediction_list.append(shifted_prediction)

        if merge_buffer_length is not None and len(object_prediction_list) > merge_buffer_length:
            postprocess_time_start = time.time()
            object_prediction_list = _apply_postprocess(
                postprocess,
                object_prediction_list,
                slice_bboxes,
                (image_h, image_w),
                slice_shape,
            )
            postprocess_time += time.time() - postprocess_time_start

    if len(slice_bboxes) > 1 and perform_standard_pred:
        prediction_result = sahi.predict.get_prediction(
            image=image,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=[image_h, image_w],
            exclude_classes_by_name=exclude_classes_by_name,
            exclude_classes_by_id=exclude_classes_by_id,
        )
        object_prediction_list.extend(_normalize_object_prediction_list(prediction_result.object_prediction_list))

    if len(object_prediction_list) > 1:
        postprocess_time_start = time.time()
        object_prediction_list = _apply_postprocess(
            postprocess,
            object_prediction_list,
            slice_bboxes,
            (image_h, image_w),
            slice_shape,
        )
        postprocess_time += time.time() - postprocess_time_start
    object_prediction_list = _normalize_object_prediction_list(object_prediction_list)

    durations_in_seconds["prediction"] = time.time() - time_start - postprocess_time
    durations_in_seconds["postprocess"] = postprocess_time

    if verbose == 2:
        print("Slicing performed in", durations_in_seconds["slice"], "seconds.")
        print("Prediction performed in", durations_in_seconds["prediction"], "seconds.")
        print("Postprocessing performed in", durations_in_seconds["postprocess"], "seconds.")

    return PredictionResult(
        image=_to_prediction_result_image(image),
        object_prediction_list=object_prediction_list,
        durations_in_seconds=durations_in_seconds,
    )


sahi.predict.get_sliced_prediction = _patched_get_sliced_prediction
get_sliced_prediction = _patched_get_sliced_prediction


def rgb_to_ir_path(rgb_path):
    dir_path, filename = os.path.split(rgb_path)
    name, ext = os.path.splitext(filename)
    parts = name.split("-")
    if parts[-1] != "1":
        return None
    parts[-1] = "2"
    return os.path.join(dir_path, "-".join(parts) + ext)


def collect_rgb_images(directory):
    valid_ext = (".jpg", ".jpeg", ".png", ".JPG")
    return [
        os.path.join(directory, file_name)
        for file_name in os.listdir(directory)
        if file_name.endswith(valid_ext) and os.path.splitext(file_name)[0].split("-")[-1] == "1"
    ]


def main():
    detection_model = CustomUltralyticsModel(
        model_path=r"C:\Users\mingzhu.li\PyCharmMiscProject\main\runs\train_new\DEYOLO_v12_o_deca_val\weights\best.pt",
        confidence_threshold=0.2,
        device="cpu",
    )

    advanced_postprocess = AdvancedPostprocessPredictions(
        match_threshold=0.5,
        use_advanced_combiner=True,
        edge_penalty_ratio=0.6,
        edge_margin=32,
        iof_threshold=0.85,
        size_ratio_threshold=0.6,
        verbose=DEBUG,
    )

    rgb_dir = r"C:\Users\mingzhu.li\PyCharmMiscProject\main\image"
    output_root = "sahi_DEYOLO_v12_o_test_clean"
    os.makedirs(output_root, exist_ok=True)

    rgb_images = collect_rgb_images(rgb_dir)
    print(f"Found {len(rgb_images)} RGB/IR pairs to process.")

    for idx, rgb_path in enumerate(rgb_images, start=1):
        print(f"\n[{idx}/{len(rgb_images)}] Processing: {os.path.basename(rgb_path)}")
        ir_path = rgb_to_ir_path(rgb_path)

        if ir_path is None or not os.path.exists(ir_path):
            print(f"Skip: missing IR image: {ir_path}")
            continue

        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        if img_rgb is None or img_ir is None:
            print("Skip: failed to read RGB or IR image.")
            continue

        img_6ch = np.concatenate((img_rgb, img_ir), axis=-1)
        result = get_sliced_prediction(
            img_6ch,
            detection_model=detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            auto_slice_resolution=False,
            perform_standard_pred=False,
            postprocess_class=advanced_postprocess,
            verbose=2 if DEBUG else 1,
        )
        result.object_prediction_list = [
            prediction for prediction in result.object_prediction_list if prediction.score.value >= 0.2
        ]

        vis_file_name = os.path.splitext(os.path.basename(rgb_path))[0]
        vis_path = os.path.join(output_root, f"{vis_file_name}.png")
        txt_path = os.path.join(output_root, f"{vis_file_name}.txt")
        reflective_light_vis = _draw_predictions_on_reflective_image(img_rgb, result.object_prediction_list)
        cv2.imwrite(vis_path, reflective_light_vis)
        _save_predictions_for_map(txt_path, result.object_prediction_list, img_rgb.shape[1], img_rgb.shape[0])

        stats = advanced_postprocess.get_stats()
        print(
            f"Saved visualization to {vis_path} and predictions to {txt_path} with "
            f"{len(result.object_prediction_list)} predictions "
            f"(edge_penalized={stats.get('edge_penalized', 0)}, "
            f"iof_filtered={stats.get('iof_filtered', 0)}, nms_removed={stats.get('nms_removed', 0)})."
        )


if __name__ == "__main__":
    main()

# - main/sahi_infer.py:342 开始主流程：加载模型、初始化后处理器、遍历 RGB/IR 图像对
#   - main/sahi_infer.py:381 把 RGB + IR 拼成 6 通道图
#   - main/sahi_infer.py:184 的自定义 get_sliced_prediction() 对 6 通道图做切片推理
#   - 每个切片内部会走 main/sahi_infer.py:93 的自定义 get_prediction()
#   - 模型真正推理在 main/sahi_infer.py:29 的 CustomUltralyticsModel.perform_inference()
#   - 所有切片结果汇总后，交给 AdvancedPostprocessPredictions 做去重/融合
#   - 最后把结果画到 RGB 图上，并通过 result.export_visuals() 导出可视化图，见 main/sahi_infer.py:398
#
#   SAHI 原生框架部分
#
#   - 你仍然在用 SAHI 的这些核心对象：
#       - UltralyticsDetectionModel 基类，见 main/sahi_infer.py:15
#       - PredictionResult，见 main/sahi_infer.py:17
#       - filter_predictions，见 main/sahi_infer.py:18
#       - get_slice_bboxes，见 main/sahi_infer.py:19
#       - PostprocessPredictions 基类，见 main/sahi_infer.py:16
#   - 也就是说，SAHI 原生框架还负责：
#       - 检测模型抽象层接口
#       - 原始预测转 ObjectPrediction
#       - 切片框网格计算
#       - 预测结果对象封装
#       - 导出可视化 export_visuals()
#
#   你的自定义处理逻辑
#
#   - 1. 6 通道输入模型适配
#       - main/sahi_infer.py:29
#       - 你继承了 UltralyticsDetectionModel，重写了 perform_inference()
#       - 你的处理包括：
#           - 接收 (H, W, 6) 的 numpy 图像
#           - 前 3 通道和后 3 通道分别做 BGR->RGB
#           - 转成 [1, 6, H, W] tensor
#           - 除以 255.0
#           - 补齐到 32 倍数后再推理
#           - 把 Ultralytics 输出整理成 SAHI 能接的格式
#   - 2. 重写 SAHI 的单图预测入口
#       - main/sahi_infer.py:93
#       - 你 monkey patch 了 sahi.predict.get_prediction
#       - 作用：
#           - 避开 SAHI 原生 read_image_as_pil() 对 6 通道图的限制
#           - 直接让 numpy 6 通道图进入模型
#           - 保留 SAHI 的 convert_original_predictions() 和 filter_predictions()
#   - 3. 重写 SAHI 的切片推理入口
#       - main/sahi_infer.py:184
#       - 你 monkey patch 了 sahi.predict.get_sliced_prediction
#       - 作用：
#           - 支持 6 通道切片图
#           - 支持传入你自己的 postprocess_class
#           - 在每个预测上补 slice_origin
#           - 把切片坐标、整图尺寸、切片尺寸传给你的高级后处理器
#   - 4. 自定义 PredictionResult 图像兼容
#       - main/sahi_infer.py:29 上面的 _to_prediction_result_image()
#       - 因为 SAHI 的 PredictionResult 内部还是会把 image 转 PIL，所以你把 6 通道图裁成前 3 通道再交给它
#       - 这一步只影响“结果对象保存的可视化底图”，不影响模型推理
#   - 5. 自定义后处理结果格式规整
#       - _normalize_object_prediction_list() 和 _apply_postprocess()
#       - 这是为了兼容 ObjectPredictionList 与普通 list，避免 export_visuals() 出错
#   - 6. 你的高级后处理
#       - 来自 main/custom_combine.py
#       - AdvancedPostprocessPredictions 是你自己的核心后处理逻辑
#       - 它不是 SAHI 原生
#
#   当前实际执行顺序
#
#   - main/sahi_infer.py:343
#       - 初始化 CustomUltralyticsModel
#   - main/sahi_infer.py:350
#       - 初始化 AdvancedPostprocessPredictions
#   - main/sahi_infer.py:367
#       - 遍历每张 -1 结尾的 RGB 图
#   - main/sahi_infer.py:369
#       - 用 rgb_to_ir_path() 找对应 -2 的 IR 图
#   - main/sahi_infer.py:381
#       - 拼成 6 通道 img_6ch
#   - main/sahi_infer.py:382
#       - 调用你重写后的 get_sliced_prediction()
#   - main/sahi_infer.py:221
#       - 内部用 SAHI 原生 get_slice_bboxes() 生成切片网格
#   - main/sahi_infer.py:254
#       - 逐片裁切图像
#   - main/sahi_infer.py:256
#       - 每片调用你重写后的 get_prediction()
#   - main/sahi_infer.py:110
#       - 实际进入 CustomUltralyticsModel.perform_inference()
#   - main/sahi_infer.py:267
#       - 给每个结果补上 slice_origin
#   - main/sahi_infer.py:295
#       - 用你的 AdvancedPostprocessPredictions 合并所有切片结果
#   - main/sahi_infer.py:396
#       - 用 RGB 图做可视化底图
#   - main/sahi_infer.py:398
#       - 调用 SAHI 原生 export_visuals() 导出结果图
#
#   你保留的 SAHI 原生能力
#
#   - 切片网格生成：get_slice_bboxes
#   - 预测结果对象封装：PredictionResult
#   - 原始检测结果转换：convert_original_predictions
#   - 类别过滤：filter_predictions
#   - 导出可视化：export_visuals
#
#   你替换掉的 SAHI 原生能力
#
#   - 原生 get_prediction
#   - 原生 get_sliced_prediction
#   - 原生 Ultralytics 推理输入处理
#   - 原生后处理策略的核心调用方式
#
#   你的关键参数值
#
#   - 模型部分，见 main/sahi_infer.py:343
#       - model_path = C:\Users\mingzhu.li\PyCharmMiscProject\main\runs\train_new\DEYOLO_v12_o_deca_val\weights\best.pt
#       - confidence_threshold = 0.2
#       - device = "cpu"
#   - 切片部分，见 main/sahi_infer.py:24 和 main/sahi_infer.py:382
#       - SLICE_HEIGHT = 1024
#       - SLICE_WIDTH = 1024
#       - OVERLAP_RATIO = 300 / 1024 ≈ 0.293
#       - perform_standard_pred = False
#   - 后处理部分，见 main/sahi_infer.py:350
#       - match_threshold = 0.5
#       - use_advanced_combiner = True
#       - edge_penalty_ratio = 0.6
#       - edge_margin = 32
#       - iof_threshold = 0.6
#       - size_ratio_threshold = 0.6
#       - verbose = DEBUG
#   - 输入输出部分
#       - 输入目录：main/sahi_infer.py:360 -> main\image_test
#       - 输出目录：main/sahi_infer.py:361 -> sahi_DEYOLO_v12_o_test_clean
#
#   你的高级后处理器里在做什么
#
#   - 在 main/custom_combine.py
#   - 主要是这几类逻辑：
#       - 边缘框惩罚 edge_penalty_ratio
#       - IoF 包含关系过滤 iof_threshold
#       - 尺寸比例判定 size_ratio_threshold
#       - 最终 NMS / IOU 合并 match_threshold
#   - 这部分本质上是在替代 SAHI 默认的简单切片结果合并，让它更贴合你原来改过源码的逻辑