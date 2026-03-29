"""
Utility functions for tensor and PIL image conversions.
"""

import numpy as np
import torch
import json
import comfy.utils
from PIL import Image
from typing import List, Union, Optional


def tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor images to PIL Images.

    Args:
        images: Tensor of shape [B, H, W, C] with values in [0, 1]

    Returns:
        List of PIL Images

    Raises:
        ValueError: If input is not a torch.Tensor or has unsupported shape
    """
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(images)}")

    # Ensure tensor is on CPU and in correct format
    images = images.cpu()

    # Handle different tensor shapes
    if images.dim() == 3:
        # Single image [H, W, C]
        images = images.unsqueeze(0)
    elif images.dim() == 2:
        # Grayscale [H, W]
        images = images.unsqueeze(0).unsqueeze(-1)

    # Convert to [0, 255] range
    if images.max() <= 1.0:
        images = images * 255.0

    images = images.clamp(0, 255).byte()

    # Convert each image in batch to PIL
    pil_images = []
    for img in images:
        img_np = img.numpy()
        if img_np.shape[-1] == 1:
            # Grayscale
            pil_img = Image.fromarray(img_np.squeeze(-1), mode='L')
        elif img_np.shape[-1] == 3:
            # RGB
            pil_img = Image.fromarray(img_np, mode='RGB')
        elif img_np.shape[-1] == 4:
            # RGBA
            pil_img = Image.fromarray(img_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported channel count: {img_np.shape[-1]}")
        pil_images.append(pil_img)

    return pil_images


def pil_to_tensor(pil_images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
    """
    Convert PIL Images to tensor format.

    Args:
        pil_images: Single PIL Image or list of PIL Images

    Returns:
        Tensor of shape [B, H, W, C] with values in [0, 1]

    Raises:
        ValueError: If input is not a PIL Image or list of PIL Images
    """
    if isinstance(pil_images, Image.Image):
        pil_images = [pil_images]

    if not isinstance(pil_images, list):
        raise ValueError(f"Expected PIL Image or list of PIL Images, got {type(pil_images)}")

    tensor_list = []
    for pil_img in pil_images:
        if not isinstance(pil_img, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(pil_img)}")

        # Convert to RGB if needed
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert to numpy array
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np)
        tensor_list.append(img_tensor)

    # Stack into batch
    images_tensor = torch.stack(tensor_list)

    return images_tensor


def masks_to_tensor(masks: Union[torch.Tensor, Image.Image, List, np.ndarray]) -> Optional[torch.Tensor]:
    """
    Convert various mask formats to tensor format.
    
    Args:
        masks: Masks in various formats (torch.Tensor, Image.Image, List, np.ndarray)
    
    Returns:
        torch.Tensor [N, H, W] with values in [0, 1], or None if conversion fails
    """
    if isinstance(masks, torch.Tensor):
        # Ensure float type and range [0, 1]
        masks = masks.float()
        # Check if tensor is not empty before calling max()
        if masks.numel() > 0 and masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        return masks.cpu()
    elif isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
        # Check if tensor is not empty before calling max()
        if masks.numel() > 0 and masks.max() > 1.0:
            masks = masks / 255.0

        # Squeeze extra channel dimension if present
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        return masks

    return masks

def _build_label_text(i, scores, labels, label):
    """Build display text from label and score for a single detection."""
    if scores is not None:
        try:
            if isinstance(scores, torch.Tensor):
                scores_flat = scores.flatten()
                if i < len(scores_flat):
                    score = scores_flat[i].item()
                    return f"{label} {score:.2f}" if label else f"id:{i} score:{score:.2f}"
                return label or f"id:{i}"
            elif isinstance(scores, (list, np.ndarray)):
                score = scores[i] if isinstance(scores[i], (int, float)) else scores[i].item()
                return f"{label} {score:.2f}" if label else f"id:{i} score:{score:.2f}"
            elif isinstance(scores, float):
                return f"{label} {scores:.2f}" if label else f"id:{i} score:{scores:.2f}"
            return label or f"id:{i}"
        except Exception as e:
            print(f"Error getting score {i}: {e}")
            return label or f"id:{i}"
    return label or f"id:{i}"


def draw_visualize_image(image, masks, scores=None, bboxs=None, alpha=0.5, stroke_width=5, font_size=24, labels=None, display_mode="masks"):

    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)[0]
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8))

    img_np = np.array(image).astype(np.float32) / 255.0

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    from PIL import ImageDraw, ImageFont
    from scipy import ndimage
    try:
        font = ImageFont.load_default().font_variant(size=font_size)
    except:
        font = ImageFont.load_default()

    np.random.seed(42)
    overlay = img_np.copy()

    show_masks = display_mode in ("masks", "both")
    show_boxes = display_mode in ("boxes", "boxes_padding", "both")
    # boxes_padding uses provided bboxs (with padding), other box modes compute from masks
    use_padded_boxes = display_mode == "boxes_padding" and bboxs is not None

    text_info_list = []
    # Pre-generate colors for consistent coloring across modes
    colors = [np.random.rand(3) for _ in range(len(masks_np))]

    num_masks = len(masks_np)
    pbar = comfy.utils.ProgressBar(num_masks)

    for i, mask in enumerate(masks_np):
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        if mask.shape != img_np.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        color = colors[i]
        stroke_color = color * 0.4

        if show_masks:
            # Mask fill + morphological stroke
            binary_mask = (mask > 0.5).astype(np.uint8)
            dilated = ndimage.binary_dilation(binary_mask, iterations=stroke_width).astype(np.float32)
            stroke_mask = dilated - binary_mask
            for c in range(3):
                overlay[:, :, c] = np.where(stroke_mask > 0.5, stroke_color[c], overlay[:, :, c])
            for c in range(3):
                overlay[:, :, c] = np.where(mask > 0.5, overlay[:, :, c] * (1 - alpha) + color[c] * alpha, overlay[:, :, c])

        # boxes_padding mode uses provided bboxs (with padding applied),
        # all other modes compute bounding box from mask pixels
        has_box = False
        if use_padded_boxes and i < len(bboxs):
            box = bboxs[i]
            x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            has_box = True
        else:
            mask_coords = np.argwhere(mask > 0.5)
            if len(mask_coords) > 0:
                y_min = int(mask_coords[:, 0].min())
                y_max = int(mask_coords[:, 0].max())
                x_min = int(mask_coords[:, 1].min())
                x_max = int(mask_coords[:, 1].max())
                has_box = True

        if has_box:
            stroke_color_int = tuple((stroke_color * 255).astype(int).tolist())
            color_int = tuple((color * 255).astype(int).tolist())

            label = None
            if labels is not None and isinstance(labels, list) and i < len(labels):
                label = labels[i] if labels[i] else None

            text = _build_label_text(i, scores, labels, label)

            # Label position: top-left of bounding box for boxes mode, center-top for masks mode
            if show_boxes and not show_masks:
                text_pos = (x_min, max(0, y_min - font_size - 8))
            else:
                x_center = (x_min + x_max) // 2
                text_pos = (x_center, max(0, y_min - font_size))

            text_info_list.append({
                'text': text,
                'position': text_pos,
                'bg_color': stroke_color_int,
                'box': (x_min, y_min, x_max, y_max) if show_boxes else None,
                'box_color': color_int,
            })

        pbar.update_absolute(i + 1, num_masks)

    result = Image.fromarray((overlay * 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    # Draw bounding box rectangles
    if show_boxes:
        for info in text_info_list:
            if info['box'] is not None:
                x1, y1, x2, y2 = info['box']
                draw.rectangle([(x1, y1), (x2, y2)], outline=info['box_color'], width=stroke_width)

    # Draw text labels on top
    padding = 8
    for info in text_info_list:
        text = info['text']
        pos = info['position']
        bbox = draw.textbbox(pos, text, font=font)
        draw.rectangle(
            [(bbox[0] - padding, bbox[1] - padding),
             (bbox[2] + padding, bbox[3] + padding)],
            fill=info['bg_color']
        )
        draw.text(pos, text, fill=(255, 255, 255), font=font)

    return result


def resize_mask(mask, shape):
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)

def join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor, invert=False):
    batch_size = min(len(image), len(alpha))
    out_images = []

    if invert:
        alpha = 1.0 - resize_mask(alpha, image.shape[1:])
    else:
        alpha = resize_mask(alpha, image.shape[1:])
    for i in range(batch_size):
        out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

    return torch.stack(out_images),

def parse_points(points_str, image_shape=None):
    """Parse point coordinates from JSON string and validate bounds.
    
    Supports two formats:
    1. {"points": [[x, y], ...], "labels": [1, 0, ...]} - Direct format with normalized coordinates
    2. [{"x": x, "y": y}, ...] - Legacy format with pixel coordinates
    
    Converts pixel coordinates to normalized coordinates (0-1 range) if image_shape is provided.

    Returns:
        tuple: (points_array, count) for format 1, or (points_array, count, validation_errors) for format 2
    """
    if not points_str or not points_str.strip():
        return None, None, []

    try:
        parsed_data = json.loads(points_str)

        # Check if it's the new format with "points" and "labels" keys
        if isinstance(parsed_data, dict) and "points" in parsed_data:
            points = parsed_data["points"]
            if not points:
                return None, None
            return points, len(points), []
        
        # Legacy format: list of point dictionaries
        if not isinstance(parsed_data, list):
            raise ValueError(f"Points must be a JSON array or object with 'points' key, got {type(parsed_data).__name__}")

        if len(parsed_data) == 0:
            return None, None, []

        points = []
        validation_errors = []

        for i, point_dict in enumerate(parsed_data):
            if not isinstance(point_dict, dict):
                err = f"Point {i} is not a dictionary"
                print(f"Warning: {err}, skipping")
                validation_errors.append(err)
                continue

            if 'x' not in point_dict or 'y' not in point_dict:
                err = f"Point {i} missing 'x' or 'y' key"
                print(f"Warning: {err}, skipping")
                validation_errors.append(err)
                continue

            try:
                x = float(point_dict['x'])
                y = float(point_dict['y'])

                # Validate coordinates are non-negative
                if x < 0 or y < 0:
                    err = f"Point {i} has negative coordinates ({x}, {y})"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

                # Normalize to 0-1 range if image shape is provided
                if image_shape is not None:
                    height, width = image_shape[1], image_shape[2]  # [batch, height, width, channels]
                    
                    # Validate within image bounds
                    if x >= width or y >= height:
                        err = f"Point {i} ({x}, {y}) outside image bounds ({width}x{height})"
                        print(f"Warning: {err}, skipping")
                        validation_errors.append(err)
                        continue
                    
                    # Normalize coordinates to [0, 1] range
                    x = x / width
                    y = y / height

                points.append([x, y])

            except (ValueError, TypeError) as e:
                err = f"Could not convert point {i} coordinates to float: {e}"
                print(f"Warning: {err}, skipping")
                validation_errors.append(err)
                continue

        if not points:
            return None, None, validation_errors

        return points, len(points), validation_errors

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in points: {str(e)}")
    except Exception as e:
        print(f"Error parsing points: {e}")
        return None, None, [str(e)]

def parse_bbox(bbox, image_shape=None):
    """Parse bounding box from BBOX type (tuple/list/dict) and validate
    
    Supports multiple formats:
    1. {"boxes": [[x, y, w, h], ...], "labels": [true/false, ...]} - Direct format with normalized coordinates
    2. KJNodes: [{'startX': x, 'startY': y, 'endX': x2, 'endY': y2}, ...]
    3. Tuple/list: (x1, y1, x2, y2) or (x, y, width, height)
    4. Dict: {'startX': x, 'startY': y, 'endX': x2, 'endY': y2}
    
    Converts pixel coordinates to normalized coordinates (0-1 range) if image_shape is provided.
    
    Returns:
        tuple: (boxes_array, count) for all formats
    """
    if bbox is None:
        return None, 0

    try:
        # Check if it's a string that needs to be parsed as JSON
        if isinstance(bbox, str):
            bbox = json.loads(bbox)
        
        # Check if it's the new format with "boxes" and "labels" keys
        if isinstance(bbox, dict) and "boxes" in bbox:
            boxes = bbox["boxes"]
            if not boxes:
                return None, 0
            return boxes, len(boxes)
        
        all_coords = []

        # Try to extract coordinates regardless of type checks
        # This handles cases where ComfyUI wraps data in unexpected ways
        if hasattr(bbox, '__iter__') and not isinstance(bbox, (str, bytes)):
            # It's some kind of sequence
            try:
                bbox_list = list(bbox)
                if len(bbox_list) == 0:
                    return None, 0

                # Check if it's a list of 4 numbers (single bbox)
                if len(bbox_list) == 4 and all(isinstance(x, (int, float)) for x in bbox_list):
                    coords = [float(x) for x in bbox_list]
                    all_coords.append(coords)
                else:
                    # Process each element as a potential bbox
                    for elem in bbox_list:
                        coords = None

                        # Try to access as dict-like (KJNodes format)
                        if hasattr(elem, '__getitem__'):
                            try:
                                x1 = float(elem['startX'])
                                y1 = float(elem['startY'])
                                x2 = float(elem['endX'])
                                y2 = float(elem['endY'])
                                coords = [x1, y1, x2, y2]
                            except (KeyError, TypeError):
                                # Not dict format, might be numeric sequence
                                pass
                        
                        # If still no coords, try as numeric sequence
                        if coords is None:
                            if hasattr(elem, '__iter__') and not isinstance(elem, (str, bytes)):
                                inner = list(elem)
                                if len(inner) == 4:
                                    coords = [float(x) for x in inner]
                        if coords is not None:
                            all_coords.append(coords)

            except Exception as e:
                raise ValueError(f"Failed to process bbox as sequence: {e}")

        # Try single dict format
        elif hasattr(bbox, '__getitem__'):
            try:
                x1 = float(bbox['startX'])
                y1 = float(bbox['startY'])
                x2 = float(bbox['endX'])
                y2 = float(bbox['endY'])
                coords = [x1, y1, x2, y2]
                all_coords.append(coords)
            except (KeyError, TypeError) as e:
                raise ValueError(f"Dictionary bbox missing required keys: {e}")

        else:
            raise ValueError(f"Unsupported bbox type: {type(bbox)}")

        if not all_coords:
            raise ValueError(
                f"Could not extract coordinates from bbox. Type: {type(bbox)}, Content: {repr(bbox)[:200]}")

        # Process and validate each bbox
        validated_coords = []
        for coords in all_coords:
            # Handle xywh format (convert to xyxy)
            x1, y1, x2, y2 = coords
            if x2 < x1 or y2 < y1:
                # Assume xywh format: (x, y, width, height)
                width, height = x2, y2
                x2 = x1 + width
                y2 = y1 + height
                coords = [x1, y1, x2, y2]

            # Validate coordinates
            if coords[0] >= coords[2]:
                raise ValueError(f"Invalid bbox: x1 ({coords[0]}) must be < x2 ({coords[2]})")
            if coords[1] >= coords[3]:
                raise ValueError(f"Invalid bbox: y1 ({coords[1]}) must be < y2 ({coords[3]})")
            if coords[0] < 0 or coords[1] < 0:
                raise ValueError(f"Bounding box coordinates must be non-negative, got x1={coords[0]}, y1={coords[1]}")
            
            # Normalize to 0-1 range if image shape is provided
            if image_shape is not None:
                height, width = image_shape[1], image_shape[2]
                
                # Validate within image bounds
                if coords[0] >= width or coords[2] > width:
                    print(f"Warning: bbox x coordinates ({coords[0]}, {coords[2]}) outside image width ({width})")
                if coords[1] >= height or coords[3] > height:
                    print(f"Warning: bbox y coordinates ({coords[1]}, {coords[3]}) outside image height ({height})")

                # Normalize coordinates to [0, 1] range
                x1 = coords[0] / width
                y1 = coords[1] / height
                x2 = coords[2] / width
                y2 = coords[3] / height
                new_coords = [
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1
                ]
            
                validated_coords.append(new_coords)
            else:
                validated_coords.append(coords)

        return validated_coords, len(validated_coords)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in bbox: {str(e)}")
    except (ValueError, TypeError) as e:
        error_msg = f"Invalid bbox: {str(e)}\n"
        error_msg += f"Input type: {type(bbox)}\n"
        error_msg += f"Input content: {repr(bbox)[:500]}"
        raise ValueError(error_msg)


if __name__ == "__main__":
    bboxes = [
        [
            159.9,
            189.5,
            317.4,
            329.3
        ]
    ]
    print(parse_bbox(bboxes, image_shape=(1, 832, 480, 3)))