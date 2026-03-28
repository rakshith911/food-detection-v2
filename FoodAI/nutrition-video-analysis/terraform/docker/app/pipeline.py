"""
Main Video Processing Pipeline
Orchestrates Florence-2, SAM2, Metric3D, and RAG for nutrition analysis
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import io
import logging
import json
import sys
import re
from datetime import datetime
import os
import subprocess
import boto3

logger = logging.getLogger(__name__)

# Initialize S3 client for uploading segmented images
s3_client = None
S3_RESULTS_BUCKET = os.environ.get('S3_RESULTS_BUCKET')
UPLOAD_SEGMENTED_IMAGES = (os.environ.get('UPLOAD_SEGMENTED_IMAGES', 'true')).strip().lower() == 'true'


class NutritionVideoPipeline:
    """
    Complete pipeline for video-based nutrition analysis
    """

    # Shared Gemini generation config — temperature=0 for deterministic/reproducible outputs
    _GEMINI_GEN_CONFIG = {"temperature": 0.0}

    def __init__(self, model_manager, config):
            """
            Initialize pipeline with models and configuration
            
            Args:
                model_manager: ModelManager instance with loaded models
                config: Settings instance with configuration
            """
            self.models = model_manager
            self.config = config
            # Use CPU when CUDA is requested but not available (e.g. on Mac / CPU-only PyTorch)
            if config.DEVICE == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
                logger.info("Pipeline: CUDA not available - using CPU for depth/tensors")
            else:
                self.device = config.DEVICE

            # Task prompts for Florence-2
            self.TASK_PROMPTS = {
                "caption": "<CAPTION>",
                "detailed_caption": "<DETAILED_CAPTION>",
                "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
                "object_detection": "<OD>",  # Direct object detection
                "hybrid_detection": "hybrid",  # Combines OD + detailed caption
                "detailed_od": "detailed_od",  # OD + basic caption for enhanced labels without hallucinations
                "vqa": "<VQA>"  # Visual Question Answering - format: <VQA> + question
            }
            
            # Calibration state
            self.calibration = {
                'pixels_per_cm': None,
                'calibrated': False,
                'reference_plane_depth_m': None  # Depth of plate/reference surface
            }
            
            # Store Florence-2 detection results for debugging
            self.florence_detections = []
    
    @staticmethod
    def _build_user_context_suffix(user_context: dict) -> str:
        """Build a prompt suffix from the user's questionnaire answers.
        NOTE: extras and hidden_ingredients are injected into results separately —
        do NOT ask Gemini to add them to visible_ingredients here (would duplicate them).
        Only pass them as context so Gemini improves its overall analysis of the image.
        """
        if not user_context:
            return ""
        lines = []
        hidden = user_context.get('hidden_ingredients', [])
        if hidden:
            items = ', '.join(
                f"{i['name']} ({i['quantity']})" if i.get('quantity') else i['name']
                for i in hidden if i.get('name')
            )
            if items:
                lines.append(
                    f"- Ingredients present but not fully visible in the image: {items}. "
                    "Use this to improve your understanding of the dish and portion estimates. "
                    "Do NOT add these as separate entries in visible_ingredients."
                )
        extras = user_context.get('extras', [])
        if extras:
            items = ', '.join(
                f"{i['name']} ({i['quantity']})" if i.get('quantity') else i['name']
                for i in extras if i.get('name')
            )
            if items:
                lines.append(
                    f"- Extras or cooking additions noted by the user: {items}. "
                    "These will be accounted for separately — do NOT add them as entries in visible_ingredients."
                )
        recipe = user_context.get('recipe_description', '').strip()
        if recipe:
            lines.append(
                f"- Recipe/menu description from user: \"{recipe}\". "
                "Use this to improve accuracy of food identification and portion estimates."
            )
        if not lines:
            return ""
        return (
            "\n\nADDITIONAL CONTEXT PROVIDED BY THE USER:\n"
            + "\n".join(lines)
            + "\n"
        )

    def process_image(self, image_path: Path, job_id: str, user_context: dict = None) -> Dict:
        """
        Process a single image (same pipeline as video, but with 1 frame)

        Args:
            image_path: Path to input image
            job_id: Unique job identifier

        Returns:
            Complete results dictionary with tracking, volumes, and nutrition
        """
        logger.info(f"[{job_id}] Starting image processing: {image_path.name}")

        # Reset per-job state so detections from previous jobs don't leak in
        self.florence_detections = []
        self.calibration = {
            'pixels_per_cm': None,
            'calibrated': False,
            'reference_plane_depth_m': None,
        }

        try:
            # Load image as a single frame
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Resize if needed
            if self.config.RESIZE_WIDTH:
                h, w = img.shape[:2]
                if w > self.config.RESIZE_WIDTH:
                    new_h = int(h * self.config.RESIZE_WIDTH / w)
                    img = cv2.resize(img, (self.config.RESIZE_WIDTH, new_h))
            
            # Convert BGR to RGB (cv2 loads as BGR, but PIL/Florence-2 expect RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frames = [img]
            logger.info(f"[{job_id}] Loaded image as single frame")

            # Step 2: Run tracking pipeline with depth
            tracking_results = self._run_tracking_pipeline(frames, job_id, user_context=user_context)

            # Step 3: Analyze nutrition
            print("🍎 Analyzing nutrition...")
            import sys
            sys.stdout.flush()
            nutrition_results = self._analyze_nutrition(tracking_results, job_id)
            print("✓ Nutrition analysis complete")
            sys.stdout.flush()

            # Step 4: Compile complete results
            final_results = {
                'job_id': job_id,
                'media_name': image_path.name,
                'media_type': 'image',
                'timestamp': datetime.utcnow().isoformat(),
                'num_frames_processed': 1,
                'calibration': self.calibration,
                'florence_detections': self.florence_detections,  # Store Florence-2 detection results
                'tracking': tracking_results,
                'nutrition': nutrition_results,
                'status': 'completed'
            }

            logger.info(f"[{job_id}] ✓ Image processing completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"[{job_id}] Image processing failed: {e}", exc_info=True)
            raise

    def process_video(self, video_path: Path, job_id: str, user_context: dict = None) -> Dict:
        """
        Main entry point - process entire video

        Args:
            video_path: Path to input video
            job_id: Unique job identifier

        Returns:
            Complete results dictionary with tracking, volumes, and nutrition
        """
        logger.info(f"[{job_id}] Starting video processing: {video_path.name}")

        # Reset per-job state so detections from previous jobs don't leak in
        self.florence_detections = []
        self.calibration = {
            'pixels_per_cm': None,
            'calibrated': False,
            'reference_plane_depth_m': None,
        }

        try:
            # Step 1: Load and prepare frames
            frames = self._load_frames(video_path)
            if not frames:
                raise ValueError("No frames loaded from video")

            logger.info(f"[{job_id}] Loaded {len(frames)} frames")

            # Step 2: Run tracking pipeline with depth (pass video_path for one-shot Gemini video)
            tracking_results = self._run_tracking_pipeline(frames, job_id, video_path=video_path, user_context=user_context)

            # Step 3: Analyze nutrition
            nutrition_results = self._analyze_nutrition(tracking_results, job_id)

            # Step 4: Compile complete results (same structure as image)
            final_results = {
                'job_id': job_id,
                'media_name': video_path.name,
                'media_type': 'video',
                'timestamp': datetime.utcnow().isoformat(),
                'num_frames_processed': len(frames),
                'calibration': self.calibration,
                'florence_detections': self.florence_detections,
                'tracking': tracking_results,
                'nutrition': nutrition_results,
                'status': 'completed'
            }

            # Step 5: Generate segmented overlay video (same directory as segmented images)
            # Use >= 1 instead of == num_frames_for_video so short videos (fewer frames than requested)
            # still get a segmented overlay generated.
            if len(frames) >= 1 and tracking_results.get('objects'):
                try:
                    self._generate_segmented_video(video_path, job_id, tracking_results)
                except Exception as e:
                    logger.warning(f"[{job_id}] Segmented video generation failed (non-fatal): {e}", exc_info=True)

            logger.info(f"[{job_id}] ✓ Processing completed successfully")
            return final_results

        except Exception as e:
            logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
            raise
    
    def _load_frames(self, video_path: Path) -> List[np.ndarray]:
        """Load frames from video. If VIDEO_NUM_FRAMES is set, enforce VIDEO_MAX_DURATION_SECONDS and load exactly that many frames evenly spaced."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        
        num_frames_to_load = getattr(self.config, "VIDEO_NUM_FRAMES", None)
        max_duration = getattr(self.config, "VIDEO_MAX_DURATION_SECONDS", None)
        
        if num_frames_to_load is not None and num_frames_to_load > 0 and max_duration is not None:
            # Use first max_duration seconds when video is longer (sample from that window)
            window_sec = min(duration_sec, max_duration)
            if duration_sec > max_duration:
                logger.warning(
                    f"Video duration {duration_sec:.1f}s exceeds maximum {max_duration}s. Sampling {num_frames_to_load} frames from first {max_duration}s only."
                )
            # Exactly N frames evenly spaced in time (same prompt logic as single image; 5 frames for no-duplicate handling)
            logger.info(f"Video: {fps:.1f}fps, {total_frames} total frames, {duration_sec:.1f}s — loading exactly {num_frames_to_load} frames (window {window_sec:.1f}s)")
            frames = []
            for i in range(num_frames_to_load):
                t_sec = (i / max(1, num_frames_to_load - 1)) * max(0.0, window_sec - 0.001)
                frame_idx = min(int(t_sec * fps), total_frames - 1) if total_frames > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(self.config.RESIZE_WIDTH * aspect_ratio)
                frame_resized = cv2.resize(frame, (self.config.RESIZE_WIDTH, new_height))
                frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(frames) != num_frames_to_load:
                if len(frames) >= 1:
                    logger.warning(
                        f"Requested {num_frames_to_load} frames but video only yielded {len(frames)} (short or low frame count). Proceeding with {len(frames)} frame(s)."
                    )
                    return frames
                raise ValueError(f"Could not load {num_frames_to_load} frames from video (got {len(frames)})")
            return frames
        
        logger.info(f"Video: {fps:.1f}fps, {total_frames} total frames")
        logger.info(f"Processing every {self.config.FRAME_SKIP} frames")
        
        frames = []
        frame_idx = 0
        frames_loaded = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.config.FRAME_SKIP == 0:
                # Resize frame
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(self.config.RESIZE_WIDTH * aspect_ratio)
                frame_resized = cv2.resize(frame, (self.config.RESIZE_WIDTH, new_height))
                frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                frames_loaded += 1
                
                # Check max_frames limit
                if self.config.MAX_FRAMES and frames_loaded >= self.config.MAX_FRAMES:
                    break
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _run_tracking_pipeline(self, frames: List[np.ndarray], job_id: str, video_path: Optional[Path] = None, user_context: dict = None) -> Dict:
        """
        Run complete tracking pipeline with depth estimation.
        When video_path is set and USE_GEMINI_VIDEO_DETECTION, runs one Gemini video call for the whole clip.
        
        Returns:
            Dict with tracked objects and volume measurements
        """
        logger.info(f"[{job_id}] Running tracking pipeline...")
        
        # Video: one-shot Gemini video only (no frame-wise detection). Image: frame-wise Gemini/Florence as needed.
        initial_video_detections = None
        use_video_detection = False
        num_frames_for_video = getattr(self.config, "VIDEO_NUM_FRAMES", None)
        use_multi_image_video = (
            video_path is not None
            and self.config.USE_GEMINI_DETECTION
            and getattr(self.config, "USE_GEMINI_VIDEO_DETECTION", True)
            and num_frames_for_video is not None
            and num_frames_for_video > 0
            and len(frames) == num_frames_for_video
        )
        is_video_one_shot_mode = (
            video_path is not None
            and self.config.USE_GEMINI_DETECTION
            and getattr(self.config, "USE_GEMINI_VIDEO_DETECTION", True)
            and len(frames) > 1
        )
        if is_video_one_shot_mode:
            if use_multi_image_video:
                print("🎬 Gemini multi-image (5 frames, no duplicates) for whole clip...")
                sys.stdout.flush()
                initial_video_detections = self._detect_objects_gemini_multi_image(frames, job_id, user_context=user_context)
            else:
                print("🎬 One-shot Gemini video detection for whole clip (no frame-wise detection)...")
                sys.stdout.flush()
                initial_video_detections = self._detect_objects_gemini_video(video_path, job_id, user_context=user_context)
            if initial_video_detections is not None:
                use_video_detection = True
                logger.info(f"[{job_id}] Using Gemini video detections for frame 0 only (one-shot only)")
            else:
                logger.warning(f"[{job_id}] Gemini video one-shot failed; falling back to frame-wise Gemini detection")
        
        # Get models (Florence only when not using Gemini for detection)
        florence_processor, florence_model = None, None
        if not self.config.USE_GEMINI_DETECTION:
            florence_processor, florence_model = self.models.florence2
        video_predictor = self.models.sam2
        # depth_anything loaded lazily on first use via self.models.depth_anything
        
        # Prepare frame directory for SAM2
        frame_dir = self.config.OUTPUT_DIR / job_id / "frames_temp"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frames
        for idx, frame in enumerate(frames):
            frame_path = frame_dir / f"{idx:05d}.jpg"
            Image.fromarray(frame).save(frame_path)
        
        # Initialize SAM2 inference state
        print("📦 Initializing SAM2 inference state...")
        sys.stdout.flush()
        try:
            inference_state = video_predictor.init_state(video_path=str(frame_dir))
            print("✓ SAM2 state initialized")
            sys.stdout.flush()
        except Exception as e:
            print(f"❌ SAM2 init failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        # Tracking state
        tracked_objects = {}
        next_object_id = 1
        colors = {}
        volume_history = {}
        video_segments = {}  # Store SAM2 masks for all frames
        sam2_to_obj_id = {}  # Map SAM2's internal IDs to our persistent obj_ids
        current_window_start = 0
        caption = None  # Store the caption from Florence-2
        # Frames collected for post-loop depth estimation: (frame_idx, frame_np, masks_dict)
        depth_candidate_frames = []
        
        # Process frames
        print(f"\n📹 Processing {len(frames)} frame(s)...")
        sys.stdout.flush()
        
        try:
            for frame_idx, frame in enumerate(frames):
                logger.debug(f"[{job_id}] Processing frame {frame_idx+1}/{len(frames)}")
                print(f"\n🖼️  Frame {frame_idx+1}/{len(frames)}")
                sys.stdout.flush()
                
                frame_pil = Image.fromarray(frame)
                
                # Periodic re-detection
                if frame_idx % self.config.DETECTION_INTERVAL == 0:
                    # Video: one-shot only — use precomputed detections at frame 0; never run frame-wise Gemini
                    detection_grams_list = []
                    detection_calories_list = []
                    if is_video_one_shot_mode and frame_idx > 0:
                        # Subsequent frames in one-shot mode — SAM2 handles tracking, skip re-detection
                        boxes = np.array([])
                        labels = []
                        detected_caption = None
                        unquantified_ingredients = []
                        logger.info(f"[{job_id}] Frame {frame_idx}: Skipping re-detection (one-shot video, SAM2 tracking)")
                    elif use_video_detection and frame_idx == 0 and initial_video_detections is not None:
                        # initial_video_detections: (boxes, labels, caption, grams_list, quantity_list [, ref_size])
                        unpacked = initial_video_detections
                        boxes_ref, labels, detected_caption = unpacked[0], unpacked[1], unpacked[2]
                        detection_grams_list = unpacked[3] if len(unpacked) > 3 else []
                        detection_quantity_list = unpacked[4] if len(unpacked) > 4 else [1] * len(labels)
                        detection_calories_list = []  # video/multi-image do not return calories yet
                        unquantified_ingredients = []
                        if not detection_grams_list:
                            detection_grams_list = [None] * len(labels)
                        if len(detection_quantity_list) != len(labels):
                            detection_quantity_list = [1] * len(labels)
                        # Scale boxes from reference size to actual frame size (ref_size = 6th elem or 1280x720)
                        h, w = frame.shape[:2]
                        ref_w = unpacked[5][0] if len(unpacked) > 5 else self._GEMINI_VIDEO_REF_W
                        ref_h = unpacked[5][1] if len(unpacked) > 5 else self._GEMINI_VIDEO_REF_H
                        scale_x = w / ref_w
                        scale_y = h / ref_h
                        boxes = np.array(boxes_ref, dtype=np.float32)
                        if len(boxes) > 0:
                            boxes[:, [0, 2]] *= scale_x
                            boxes[:, [1, 3]] *= scale_y
                            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
                        if detected_caption:
                            caption = detected_caption
                        print(f"✓ Using one-shot Gemini video detections: {len(labels)} objects")
                        sys.stdout.flush()
                        logger.info(f"[{job_id}] Frame {frame_idx}: Gemini video (one-shot) {len(boxes)} objects: {labels}")
                    else:
                        # Image or video without one-shot: frame-wise detection (Gemini image or Florence-2)
                        detection_grams_list = []
                        detection_calories_list = []
                        logger.info(f"[{job_id}] Frame {frame_idx}: Re-detecting objects...")
                        if self.config.USE_GEMINI_DETECTION:
                            print(f"🔍 Detecting objects in frame {frame_idx} (Gemini image understanding)...")
                        else:
                            print(f"🔍 Detecting objects in frame {frame_idx}... (this may take 30-60 seconds on CPU)")
                        sys.stdout.flush()
                        try:
                            if self.config.USE_GEMINI_DETECTION:
                                gemini_out = self._detect_objects_gemini(frame_pil, job_id, user_context=user_context)
                                boxes = gemini_out[0]
                                labels = gemini_out[1]
                                detected_caption = gemini_out[2]
                                unquantified_ingredients = gemini_out[3]
                                detection_grams_list = gemini_out[4]
                                detection_quantity_list = gemini_out[5]
                                detection_calories_list = gemini_out[6] if len(gemini_out) > 6 else []
                                if len(detection_calories_list) != len(labels):
                                    detection_calories_list = [None] * len(labels)
                            else:
                                boxes, labels, detected_caption, unquantified_ingredients = self._detect_objects_florence(
                                    frame_pil, florence_processor, florence_model
                                )
                                detection_grams_list = []
                                detection_quantity_list = [1] * len(labels)
                                detection_calories_list = []
                            if detected_caption:
                                caption = detected_caption
                            print(f"✓ Detection complete: found {len(boxes)} objects")
                            sys.stdout.flush()
                        except Exception as e:
                            print(f"❌ Detection failed: {e}")
                            import traceback
                            traceback.print_exc()
                            sys.stdout.flush()
                            raise
                        logger.info(f"[{job_id}] Frame {frame_idx}: Detected {len(boxes)} objects: {labels}")
                    
                    # Use Gemini to format VQA answer and filter non-food items (only when using Florence-2)
                    if not self.config.USE_GEMINI_DETECTION and self.config.GEMINI_API_KEY and len(labels) > 0:
                        filtered_boxes, filtered_labels, formatted_answer = self._format_and_filter_with_gemini(
                            boxes, labels, detected_caption, job_id, frame_idx
                        )
                        boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])
                        labels = filtered_labels
                        if formatted_answer:
                            detected_caption = formatted_answer  # Update caption with formatted version
                        logger.info(f"[{job_id}] Frame {frame_idx}: After Gemini filtering: {len(boxes)} food items: {labels}")
                    
                    # Store detection results for debugging (Gemini or Florence-2)
                    detection_info = {
                        'frame_idx': frame_idx,
                        'caption': caption,
                        'detections': [
                            {
                                'label': label,
                                'box': box.tolist() if hasattr(box, 'tolist') else list(box),
                                'box_area': float((box[2] - box[0]) * (box[3] - box[1]))
                            }
                            for box, label in zip(boxes, labels)
                        ],
                        'total_detected': len(boxes),
                        'unquantified_ingredients': unquantified_ingredients  # Ingredients detected but not localized
                    }
                    self.florence_detections.append(detection_info)
                    
                    if len(boxes) > 0:
                        # Match new detections to existing objects using spatial overlap (IoU)
                        # High IoU = same object, Low IoU = new object
                        # Use greedy 1-to-1 matching: each object matches to at most one detection
                        matched_mapping = {}  # Maps existing_obj_id -> new_detection_idx
                        matched_new_indices = set()  # Track which new detections are already matched
                        unmatched_new = list(range(len(boxes)))  # Indices of new detections not matched
                        
                        if tracked_objects:
                            # Calculate IoU matrix
                            iou_matrix = []
                            for obj_id, obj_data in tracked_objects.items():
                                if 'box' in obj_data:
                                    row = []
                                    for new_box in boxes:
                                        iou = self._calculate_iou(new_box, obj_data['box'])
                                        row.append(iou)
                                    iou_matrix.append((obj_id, row))
                            
                            # Greedy matching: match highest IoU pairs first
                            while iou_matrix:
                                # Find the highest IoU across all (object, detection) pairs
                                best_iou = 0.0
                                best_obj_id = None
                                best_new_idx = None
                                
                                for obj_id, iou_row in iou_matrix:
                                    for new_idx, iou in enumerate(iou_row):
                                        if new_idx not in matched_new_indices and iou > best_iou:
                                            best_iou = iou
                                            best_obj_id = obj_id
                                            best_new_idx = new_idx
                                
                                # If best IoU > 0.5, match them
                                if best_iou > 0.5 and best_obj_id is not None:
                                    matched_mapping[best_obj_id] = best_new_idx
                                    matched_new_indices.add(best_new_idx)
                                    if best_new_idx in unmatched_new:
                                        unmatched_new.remove(best_new_idx)
                                    logger.info(f"[{job_id}] Frame {frame_idx}: Matched '{labels[best_new_idx]}' to existing ID{best_obj_id} ('{tracked_objects[best_obj_id]['label']}') with IoU={best_iou:.2f}")
                                    
                                    # Remove matched object from matrix
                                    iou_matrix = [(oid, row) for oid, row in iou_matrix if oid != best_obj_id]
                                else:
                                    break  # No more good matches
                        
                        logger.info(f"[{job_id}] Frame {frame_idx}: Matched {len(matched_mapping)} objects, {len(unmatched_new)} new objects")
                        
                        # Reset SAM2 state
                        inference_state = video_predictor.init_state(video_path=str(frame_dir))
                        video_segments = {}  # Reset video segments when SAM2 resets
                        sam2_to_obj_id = {}  # Reset SAM2 ID mapping
                        current_window_start = frame_idx
                        
                        # Update tracked objects
                        boxes_to_add = []
                        ids_to_add = []
                        
                        # Matched objects (update label to latest detection, keep same ID)
                        for old_id, new_idx in matched_mapping.items():
                            old_label = tracked_objects[old_id]['label']
                            new_label = labels[new_idx]
                            if old_label != new_label:
                                logger.info(f"[{job_id}] Frame {frame_idx}: Updating label for ID{old_id}: '{old_label}' → '{new_label}'")
                            tracked_objects[old_id]['box'] = boxes[new_idx]
                            tracked_objects[old_id]['label'] = new_label
                            tracked_objects[old_id]['last_seen_frame'] = frame_idx
                            if detection_grams_list and new_idx < len(detection_grams_list) and detection_grams_list[new_idx] is not None:
                                tracked_objects[old_id]['gemini_grams'] = float(detection_grams_list[new_idx])
                            if detection_quantity_list and new_idx < len(detection_quantity_list):
                                try:
                                    tracked_objects[old_id]['gemini_quantity'] = max(1, int(detection_quantity_list[new_idx]))
                                except (TypeError, ValueError):
                                    pass
                            if detection_calories_list and new_idx < len(detection_calories_list) and detection_calories_list[new_idx] is not None:
                                tracked_objects[old_id]['gemini_kcal'] = float(detection_calories_list[new_idx])
                            boxes_to_add.append(boxes[new_idx])
                            ids_to_add.append(old_id)
                        
                        # New objects (no spatial overlap with existing - these are NEW food items)
                        for new_idx in unmatched_new:
                            obj_id = next_object_id
                            next_object_id += 1
                            
                            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
                            colors[obj_id] = color
                            gemini_grams = None
                            if detection_grams_list and new_idx < len(detection_grams_list) and detection_grams_list[new_idx] is not None:
                                gemini_grams = float(detection_grams_list[new_idx])
                            quantity = 1
                            if detection_quantity_list and new_idx < len(detection_quantity_list):
                                try:
                                    quantity = max(1, int(detection_quantity_list[new_idx]))
                                except (TypeError, ValueError):
                                    quantity = 1
                            gemini_kcal = None
                            if detection_calories_list and new_idx < len(detection_calories_list) and detection_calories_list[new_idx] is not None:
                                gemini_kcal = float(detection_calories_list[new_idx])
                            tracked_objects[obj_id] = {
                                'box': boxes[new_idx],
                                'label': labels[new_idx],
                                'color': color,
                                'first_seen_frame': frame_idx,
                                'last_seen_frame': frame_idx,
                                'gemini_grams': gemini_grams,
                                'gemini_quantity': quantity,
                                'gemini_kcal': gemini_kcal
                            }
                            
                            boxes_to_add.append(boxes[new_idx])
                            ids_to_add.append(obj_id)
                            logger.info(f"[{job_id}] Frame {frame_idx}: Added NEW object ID{obj_id} ('{labels[new_idx]}') - no spatial overlap with existing objects")
                        
                        # Add objects to SAM2 with sequential SAM2 IDs (1, 2, 3...)
                        successfully_added = []
                        sam2_id = 1  # SAM2 uses sequential IDs starting from 1
                        for i, obj_id in enumerate(ids_to_add):
                            box = boxes_to_add[i]
                            label = tracked_objects[obj_id]['label']
                            
                            # Validate box coordinates
                            x1, y1, x2, y2 = box
                            if x2 <= x1 or y2 <= y1:
                                logger.error(f"[{job_id}] Frame {frame_idx}: Invalid box for object ID{obj_id} ({label}): {box}")
                                continue
                            
                            # Ensure box is within frame bounds
                            h, w = frame.shape[:2]
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(x1+1, min(x2, w))
                            y2 = max(y1+1, min(y2, h))
                            # SAM2 expects box in format [[x1, y1], [x2, y2]] not [x1, y1, x2, y2]
                            box_sam = np.array([[[x1, y1], [x2, y2]]])
                            
                            logger.info(f"[{job_id}] Frame {frame_idx}: Adding object ID{obj_id} ({label}) to SAM2 as SAM2_ID{sam2_id} with box: {box_sam[0]}")
                            try:
                                video_predictor.add_new_points_or_box(
                                    inference_state=inference_state,
                                    frame_idx=frame_idx - current_window_start,
                                    obj_id=sam2_id,  # Use SAM2's sequential ID
                                    box=box_sam,
                                )
                                sam2_to_obj_id[sam2_id] = obj_id  # Map SAM2 ID to our persistent ID
                                successfully_added.append(obj_id)
                                logger.info(f"[{job_id}] Frame {frame_idx}: ✅ Successfully added object ID{obj_id} ({label}) to SAM2")
                                sam2_id += 1
                            except Exception as e:
                                logger.error(f"[{job_id}] Frame {frame_idx}: ❌ FAILED to add object ID{obj_id} ({label}) to SAM2: {e}", exc_info=True)
                        logger.info(f"[{job_id}] Frame {frame_idx}: Added {len(successfully_added)}/{len(ids_to_add)} objects to SAM2. Successfully added IDs: {successfully_added}")
                        
                        # Get masks for the current detection frame only (optimization)
                        relative_idx = 0  # Detection happens at start of window
                        logger.info(f"[{job_id}] Frame {frame_idx}: Getting SAM2 masks for detection frame...")
                        try:
                            out_frame_idx, sam2_obj_ids, out_mask_logits = video_predictor.infer_single_frame(
                                inference_state, relative_idx
                            )
                            # Map SAM2's IDs back to our persistent obj_ids
                            video_segments[relative_idx] = {}
                            for i, sam2_id in enumerate(sam2_obj_ids):
                                if sam2_id in sam2_to_obj_id:
                                    obj_id = sam2_to_obj_id[sam2_id]
                                    video_segments[relative_idx][obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                                else:
                                    logger.warning(f"[{job_id}] Frame {frame_idx}: SAM2 returned ID{sam2_id} not in mapping!")
                            logger.info(f"[{job_id}] Frame {frame_idx}: Got masks for {len(video_segments[relative_idx])} objects (obj_ids: {list(video_segments[relative_idx].keys())})")
                        except Exception as e:
                            logger.error(f"[{job_id}] Frame {frame_idx}: SAM2 inference failed: {e}")
                        
                        # Collect masks for this detection frame and upload segmented overlay
                        if relative_idx in video_segments:
                            masks_dict = {}
                            for obj_id in video_segments[relative_idx]:
                                if obj_id in tracked_objects:
                                    mask = video_segments[relative_idx][obj_id][0]
                                    masks_dict[obj_id] = mask

                            if masks_dict:
                                self._save_segmentation_masks(frame, masks_dict, tracked_objects, frame_idx, job_id)
                                # Record frame + masks for depth processing after the loop
                                depth_candidate_frames.append((frame_idx, frame, masks_dict))

            # No additional processing needed — depth and volume are computed after the loop
            
            print(f"✓ Frame {frame_idx} processing complete")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        # Final deduplication: merge tracked objects that are duplicates
        tracked_objects = self._deduplicate_tracked_objects(tracked_objects, volume_history)

        # ----------------------------------------------------------------
        # Post-loop: Depth Anything V2 on up to DEPTH_NUM_FRAMES frames
        # then Gemini volume estimation pass
        # ----------------------------------------------------------------
        gemini_volume_map = {}  # {label_lower: volume_ml}
        if depth_candidate_frames and tracked_objects and self.config.GEMINI_API_KEY:
            num_depth = getattr(self.config, "DEPTH_NUM_FRAMES", 3)
            # Pick evenly-spaced frames from candidates
            step = max(1, len(depth_candidate_frames) // num_depth)
            selected = depth_candidate_frames[::step][:num_depth]

            depth_maps = []
            best_frame = None
            best_masks = None
            best_frame_idx = None
            for fidx, fframe, fmasks in selected:
                try:
                    print(f"  Running Depth Anything V2 on frame {fidx}...")
                    sys.stdout.flush()
                    dmap = self._estimate_depth_anything(fframe)
                    depth_maps.append(dmap)
                    if best_frame is None:
                        best_frame = fframe
                        best_masks = fmasks
                        best_frame_idx = fidx
                except Exception as e:
                    logger.warning(f"[{job_id}] Depth Anything failed on frame {fidx}: {e}")

            if depth_maps and best_frame is not None:
                # Average depth maps across selected frames for stability
                avg_depth = np.mean(np.stack(depth_maps, axis=0), axis=0)

                # Create masked depth map image
                depth_image = self._create_masked_depth_map(
                    best_frame, avg_depth, best_masks, tracked_objects
                )
                # Upload masked depth map to S3
                self._upload_depth_map_to_s3(job_id, best_frame_idx, depth_image)

                # Use standard plate diameter as scale reference for Gemini prompt
                plate_diameter_cm = self.config.REFERENCE_PLATE_DIAMETER_CM

                # Second Gemini call: volume estimation
                gemini_volume_map = self._estimate_volume_from_depth_with_gemini(
                    best_frame, depth_image, tracked_objects,
                    plate_diameter_cm, job_id, user_context
                )
        else:
            if not depth_candidate_frames:
                logger.info(f"[{job_id}] No depth candidate frames — skipping depth/volume pass")
            elif not self.config.GEMINI_API_KEY:
                logger.info(f"[{job_id}] No Gemini API key — skipping volume estimation pass")

        # Store Gemini volume estimates on tracked objects
        for obj_id, obj_data in tracked_objects.items():
            label_lower = obj_data.get("label", "").lower()
            vol = gemini_volume_map.get(label_lower)
            if vol is None:
                # Try partial match
                for k, v in gemini_volume_map.items():
                    if k in label_lower or label_lower in k:
                        vol = v
                        break
            if vol is not None and vol > 0:
                obj_data["gemini_volume_ml"] = float(vol)
                logger.info(f"[{job_id}] ID{obj_id} ({obj_data['label']}): Gemini volume = {vol:.1f} ml")

        # Compile results
        print("Compiling results...")
        sys.stdout.flush()
        results = {
            'objects': {},
            'total_objects': len(tracked_objects),
            'caption': caption  # Include the Florence-2 caption
        }
        
        # Compile results for ALL objects that have volume history (not just current tracked_objects)
        # This ensures we don't lose objects from previous SAM2 windows
        objects_with_volume = set()
        items_for_validation = []  # Collect items with calculated volumes for batch validation
        
        for obj_id in volume_history.keys():
            history = volume_history[obj_id]
            if len(history) > 0:
                # Get label from tracked_objects, or from history if not in current tracking
                if obj_id in tracked_objects:
                    label = tracked_objects[obj_id]['label']
                else:
                    # Object from previous window - need to retrieve label
                    # For now, mark as "Unknown" but this should be fixed by accumulation
                    label = f"Unknown_{obj_id}"
                    logger.warning(f"[{job_id}] Object ID{obj_id} has volume history but is not in tracked_objects")
                
                volumes = [h['volume_ml'] for h in history]
                heights = [h['height_cm'] for h in history]
                areas = [h['area_cm2'] for h in history]
                diameters = [h.get('diameter_cm', 0) for h in history]  # Get stored diameter
                
                max_volume = float(max(volumes))
                max_height = float(max(heights))
                max_area = float(max(areas))
                max_diameter = float(max(diameters)) if diameters else 0.0
                
                gemini_grams_g = None
                gemini_quantity = 1
                gemini_kcal = None
                if obj_id in tracked_objects:
                    g = tracked_objects[obj_id].get('gemini_grams')
                    if g is not None and g > 0:
                        gemini_grams_g = float(g)
                    q = tracked_objects[obj_id].get('gemini_quantity')
                    if q is not None and q >= 1:
                        gemini_quantity = int(q)
                    k = tracked_objects[obj_id].get('gemini_kcal')
                    if k is not None and k > 0:
                        gemini_kcal = float(k)
                # Store for batch validation
                items_for_validation.append({
                    'obj_id': obj_id,
                    'label': label,
                    'calculated_volume_ml': max_volume,
                    'height_cm': max_height,
                    'area_cm2': max_area,
                    'diameter_cm': max_diameter,
                    'volumes': volumes,
                    'heights': heights,
                    'areas': areas,
                    'gemini_grams_g': gemini_grams_g,
                    'gemini_quantity': gemini_quantity,
                    'gemini_kcal': gemini_kcal
                })
                
                objects_with_volume.add(obj_id)
        
        # Include ALL tracked objects, even if they don't have volume calculations
        # Collect untracked items for batch estimation
        untracked_items = []
        for obj_id, obj_data in tracked_objects.items():
            if obj_id not in objects_with_volume:
                label = obj_data['label']
                box = obj_data['box']
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                px_per_cm = self.calibration['pixels_per_cm'] or self.config.DEFAULT_PIXELS_PER_CM
                area_cm2 = box_area / (px_per_cm ** 2)
                g = obj_data.get('gemini_grams')
                gemini_grams_g = float(g) if g is not None and g > 0 else None
                q = obj_data.get('gemini_quantity')
                gemini_quantity = max(1, int(q)) if q is not None and q >= 1 else 1
                k = obj_data.get('gemini_kcal')
                gemini_kcal = float(k) if k is not None and k > 0 else None
                untracked_items.append({
                    'obj_id': obj_id,
                    'label': label,
                    'area_cm2': area_cm2,
                    'box': box,
                    'gemini_grams_g': gemini_grams_g,
                    'gemini_quantity': gemini_quantity,
                    'gemini_kcal': gemini_kcal
                })
        
        # Batch process: Validate calculated volumes + Estimate untracked volumes in ONE Gemini call
        if self.config.GEMINI_API_KEY and (items_for_validation or untracked_items):
            validated_and_estimated = self._batch_validate_and_estimate_volumes_with_gemini(
                items_for_validation, untracked_items, job_id
            )
            validated_volumes = validated_and_estimated.get('validated', {})
            estimated_volumes = validated_and_estimated.get('estimated', {})
        else:
            # Fallback: no validation, simple estimation
            validated_volumes = {item['obj_id']: item['calculated_volume_ml'] for item in items_for_validation}
            estimated_volumes = {item['obj_id']: item['area_cm2'] * 2.0 for item in untracked_items}
        
        # Add items with validated volumes to results
        for item in items_for_validation:
            obj_id = item['obj_id']
            label = item['label']
            validated_volume = validated_volumes.get(obj_id, item['calculated_volume_ml'])
            
            if validated_volume != item['calculated_volume_ml']:
                logger.info(f"[{job_id}] ✓ Gemini adjusted volume for '{label}': {item['calculated_volume_ml']:.1f}ml → {validated_volume:.1f}ml")
            
            stats = {
                'max_volume_ml': float(validated_volume),
                'median_volume_ml': float(np.median(item['volumes'])),
                'mean_volume_ml': float(np.mean(item['volumes'])),
                'max_height_cm': float(max(item['heights'])),
                'max_area_cm2': float(max(item['areas'])),
                'num_frames': len(item['volumes'])
            }
            if item.get('gemini_grams_g') is not None and item['gemini_grams_g'] > 0:
                stats['gemini_grams_g'] = float(item['gemini_grams_g'])
            if item.get('gemini_quantity') is not None and item['gemini_quantity'] >= 1:
                stats['quantity'] = int(item['gemini_quantity'])
            else:
                stats['quantity'] = 1
            if item.get('gemini_kcal') is not None and item['gemini_kcal'] > 0:
                stats['gemini_kcal'] = float(item['gemini_kcal'])
            obj_entry = {'label': label, 'statistics': stats}
            if obj_id in tracked_objects:
                obj_entry['obj_id'] = obj_id
                box = tracked_objects[obj_id].get('box')
                if box is not None:
                    obj_entry['box'] = box.tolist() if hasattr(box, 'tolist') else list(box)
                gvol = tracked_objects[obj_id].get('gemini_volume_ml')
                if gvol is not None and gvol > 0:
                    obj_entry['gemini_volume_ml'] = float(gvol)
            results['objects'][f"ID{obj_id}_{label}"] = obj_entry

        # Add untracked items with estimated volumes to results
        for item in untracked_items:
            obj_id = item['obj_id']
            label = item['label']
            area_cm2 = item['area_cm2']
            estimated_volume_ml = estimated_volumes.get(obj_id, area_cm2 * 2.0)
            
            logger.info(f"[{job_id}] Object ID{obj_id} ('{label}') detected but no volume calculated - using estimated volume {estimated_volume_ml:.1f}ml")
            
            stats = {
                'max_volume_ml': float(estimated_volume_ml),
                'median_volume_ml': float(estimated_volume_ml),
                'mean_volume_ml': float(estimated_volume_ml),
                'max_height_cm': 2.0,  # Default estimate
                'max_area_cm2': float(area_cm2),
                'num_frames': 1,
                'estimated': True,  # Flag to indicate this is an estimate
                'estimation_method': 'gemini' if self.config.GEMINI_API_KEY else 'fallback'
            }
            if item.get('gemini_grams_g') is not None and item['gemini_grams_g'] > 0:
                stats['gemini_grams_g'] = float(item['gemini_grams_g'])
            if item.get('gemini_quantity') is not None and item['gemini_quantity'] >= 1:
                stats['quantity'] = int(item['gemini_quantity'])
            else:
                stats['quantity'] = 1
            if item.get('gemini_kcal') is not None and item['gemini_kcal'] > 0:
                stats['gemini_kcal'] = float(item['gemini_kcal'])
            obj_entry = {'label': label, 'statistics': stats}
            if obj_id in tracked_objects:
                obj_entry['obj_id'] = obj_id
                box = tracked_objects[obj_id].get('box')
                if box is not None:
                    obj_entry['box'] = box.tolist() if hasattr(box, 'tolist') else list(box)
                gvol = tracked_objects[obj_id].get('gemini_volume_ml')
                if gvol is not None and gvol > 0:
                    obj_entry['gemini_volume_ml'] = float(gvol)
            results['objects'][f"ID{obj_id}_{label}"] = obj_entry
        
        logger.info(f"[{job_id}] Tracked {len(results['objects'])} objects across all frames ({len(objects_with_volume)} with calculated volumes, {len(results['objects']) - len(objects_with_volume)} with estimated volumes)")
        results['total_objects'] = len(results['objects'])
        return results
    
    def _deduplicate_tracked_objects(self, tracked_objects, volume_history):
        """Remove duplicate tracked objects with same label and overlapping boxes"""
        if len(tracked_objects) <= 1:
            return tracked_objects
        
        # Normalize labels
        def normalize_label(label):
            label_lower = label.lower().strip()
            for article in ['a ', 'an ', 'the ']:
                if label_lower.startswith(article):
                    label_lower = label_lower[len(article):].strip()
            if label_lower.endswith('s') and label_lower not in ['glass', 'glasses', 'fries', 'nuggets']:
                label_lower = label_lower[:-1]
            return label_lower
        
        # Convert to list for easier processing
        obj_list = [(obj_id, obj_data) for obj_id, obj_data in tracked_objects.items()]
        keep = [True] * len(obj_list)
        
        # Check each pair for duplicates
        for i in range(len(obj_list)):
            if not keep[i]:
                continue
            
            obj_id_i, obj_data_i = obj_list[i]
            box_i = obj_data_i['box']
            label_i_norm = normalize_label(obj_data_i['label'])
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            
            for j in range(i + 1, len(obj_list)):
                if not keep[j]:
                    continue
                
                obj_id_j, obj_data_j = obj_list[j]
                box_j = obj_data_j['box']
                label_j_norm = normalize_label(obj_data_j['label'])
                
                # Only check duplicates if labels match
                if label_i_norm != label_j_norm:
                    continue
                
                # Check IoU and center distance
                iou = self._calculate_iou(box_i, box_j)
                center_i = np.array([(box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2])
                center_j = np.array([(box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2])
                center_dist = np.linalg.norm(center_i - center_j)
                avg_size = np.mean([box_i[2] - box_i[0], box_i[3] - box_i[1], box_j[2] - box_j[0], box_j[3] - box_j[1]])
                
                # If overlapping or very close, remove duplicate (keep larger one)
                if iou > 0.2 or center_dist < avg_size * 0.5:
                    area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                    if area_i >= area_j:
                        # Keep i, remove j
                        keep[j] = False
                        # Merge volume history from j into i
                        if obj_id_j in volume_history and obj_id_i in volume_history:
                            volume_history[obj_id_i].extend(volume_history[obj_id_j])
                            logger.info(f"Merged volume history from ID{obj_id_j} into ID{obj_id_i} (duplicate '{obj_data_j['label']}')")
                    else:
                        # Keep j, remove i
                        keep[i] = False
                        # Merge volume history from i into j
                        if obj_id_i in volume_history and obj_id_j in volume_history:
                            volume_history[obj_id_j].extend(volume_history[obj_id_i])
                            logger.info(f"Merged volume history from ID{obj_id_i} into ID{obj_id_j} (duplicate '{obj_data_i['label']}')")
                        break
        
        # Build result dict and clean up volume_history for removed objects
        result = {}
        removed_ids = []
        for i, (obj_id, obj_data) in enumerate(obj_list):
            if keep[i]:
                result[obj_id] = obj_data
            else:
                removed_ids.append(obj_id)
        
        # Remove volume history for objects that were deduplicated
        for obj_id in removed_ids:
            if obj_id in volume_history:
                del volume_history[obj_id]
                logger.debug(f"Removed volume history for deduplicated object ID{obj_id}")
        
        return result
    
    def _detect_objects_gemini(self, image_pil, job_id: str, user_context: dict = None):
        """
        Detect food objects using Gemini image understanding (same structure as gemini/test_gemini_analysis).
        Returns (boxes, labels, caption, unquantified_ingredients) for pipeline compatibility.
        """
        import sys
        sys.stdout.flush()
        if not self.config.GEMINI_API_KEY:
            logger.warning("[Gemini detection] GEMINI_API_KEY not set; returning no detections")
            return np.array([]), [], "", [], [], []
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.GEMINI_API_KEY)
        except Exception as e:
            logger.warning(f"[Gemini detection] Failed to init Gemini: {e}")
            return np.array([]), [], "", [], [], []
        img_width, img_height = image_pil.size
        prompt = (
            "Analyze this food image in detail. Provide a comprehensive analysis including:\n"
            "1. MAIN DISH/FOOD ITEM: Primary food name, cuisine type, cooking method.\n"
            "2. VISIBLE INGREDIENTS WITH LOCATIONS: List all visible ingredients/components (garnishes, sides, sauces). "
            "For each visible food item provide bounding box [x_min, y_min, x_max, y_max], estimated_quantity_grams, and quantity (count).\n"
            f"Image dimensions: {img_width} x {img_height} pixels. Bounding boxes in pixels (0 to width/height).\n"
            "estimated_quantity_grams: TOTAL edible mass in grams for that item. When there are multiple identical pieces (e.g. 6 kiwi slices, grapes, almonds), use ONE entry with quantity set to the count and estimated_quantity_grams as the TOTAL mass for all of them.\n"
            "quantity: integer count of identical items. For small countable items (nuts, almonds, berries, grapes, cherry tomatoes, etc.) COUNT EVERY visible piece and set quantity to that exact total (e.g. 15 almonds -> quantity 15, one entry with a bounding box around the whole portion; ~1g per almond). Do not underestimate: if you see 15 almonds, use quantity 15.\n"
            "Use realistic typical weights (e.g. one fish fillet 80–120g, sauce 40–80g, 15 almonds ~15g total, six kiwi slices ~120g). "
            "Different ingredients should have different estimated_quantity_grams; for the same ingredient with many pieces use one entry and total mass.\n"
            "3. INGREDIENT BREAKDOWN, 4. NUTRITIONAL INFORMATION, 5. ADDITIONAL NOTES.\n\n"
            "Format as JSON: main_food_item, cuisine_type, cooking_method, "
            "visible_ingredients (array of {name, bounding_box [x_min,y_min,x_max,y_max], estimated_quantity_grams, quantity, estimated_total_kcal}), "
            "ingredient_breakdown, nutritional_info, allergens, dietary_tags, additional_notes.\n"
            "estimated_total_kcal: total calories in kcal for that food item (use typical values, e.g. 15 almonds ~90 kcal, fish fillet ~120 kcal).\n"
            "Example: [{\"name\": \"fish fillet\", \"bounding_box\": [100,50,300,250], \"estimated_quantity_grams\": 95, \"quantity\": 1, \"estimated_total_kcal\": 120}, {\"name\": \"almonds\", \"bounding_box\": [50,400,280,520], \"estimated_quantity_grams\": 15, \"quantity\": 15, \"estimated_total_kcal\": 90}]. "
            "Output only valid JSON (you may wrap in ```json)."
        )
        prompt += self._build_user_context_suffix(user_context)
        # Try multiple models (404 if model name not available in this API version)
        gemini_models_try = ["gemini-pro-latest", "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash"]
        response_text = ""
        for model_name in gemini_models_try:
            try:
                print(f"  → Calling Gemini for food detection ({model_name})...")
                sys.stdout.flush()
                gemini_model = genai.GenerativeModel(model_name, generation_config=self._GEMINI_GEN_CONFIG)
                response = gemini_model.generate_content([prompt, image_pil])
                response_text = response.text or ""
                if response_text:
                    break
            except Exception as e:
                logger.warning(f"[Gemini detection] {model_name} failed: {e}")
                continue
        if not response_text:
            logger.warning("[Gemini detection] All models failed; returning no detections")
            return np.array([]), [], "", [], [], []
        # Parse JSON from response
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end] if json_start >= 0 else ""
            if not json_str:
                return np.array([]), [], "", [], [], []
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[Gemini detection] JSON parse failed: {e}")
            return np.array([]), [], "", [], [], []
        visible = data.get("visible_ingredients") or []
        boxes = []
        labels = []
        grams_list = []
        quantity_list = []
        calories_list = []
        # Minimum box area to filter out Gemini placeholder boxes (e.g. [0,0,1,1])
        MIN_BOX_AREA = 100
        for ing in visible:
            bbox = ing.get("bounding_box")
            name = (ing.get("name") or "").strip()
            if not name or not bbox or len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0, min(float(x_min), img_width))
            y_min = max(0, min(float(y_min), img_height))
            x_max = max(0, min(float(x_max), img_width))
            y_max = max(0, min(float(y_max), img_height))
            if x_max <= x_min or y_max <= y_min:
                continue
            # Drop tiny placeholder boxes Gemini creates for duplicate/ghost entries
            if (x_max - x_min) * (y_max - y_min) < MIN_BOX_AREA:
                continue
            g = ing.get("estimated_quantity_grams")
            try:
                grams_list.append(float(g) if g is not None else None)
            except (TypeError, ValueError):
                grams_list.append(None)
            q = ing.get("quantity")
            try:
                quantity_list.append(max(1, int(q)) if q is not None else 1)
            except (TypeError, ValueError):
                quantity_list.append(1)
            k = ing.get("estimated_total_kcal")
            try:
                calories_list.append(float(k) if k is not None and float(k) > 0 else None)
            except (TypeError, ValueError):
                calories_list.append(None)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(name)

        # Deduplicate: if both "X (added)" and "X" exist, drop the plain "X"
        # Gemini sometimes returns the same ingredient twice — once labelled and once as a ghost
        labelled_bases = {
            lbl.lower().replace('(added)', '').replace('(hidden)', '').strip()
            for lbl in labels
            if '(added)' in lbl.lower() or '(hidden)' in lbl.lower()
        }
        if labelled_bases:
            keep_indices = []
            for i, lbl in enumerate(labels):
                normalized = lbl.lower().replace('(added)', '').replace('(hidden)', '').strip()
                is_plain_duplicate = (
                    '(added)' not in lbl.lower()
                    and '(hidden)' not in lbl.lower()
                    and normalized in labelled_bases
                )
                if not is_plain_duplicate:
                    keep_indices.append(i)
            boxes        = [boxes[i]        for i in keep_indices]
            labels       = [labels[i]       for i in keep_indices]
            grams_list   = [grams_list[i]   for i in keep_indices]
            quantity_list= [quantity_list[i] for i in keep_indices]
            calories_list= [calories_list[i] for i in keep_indices]
        caption = data.get("main_food_item") or ""
        if data.get("additional_notes"):
            caption = f"{caption}. {data['additional_notes']}" if caption else data["additional_notes"]

        # Inject hidden_ingredients and extras from user_context as ground-truth items.
        # These are not visually detectable so Gemini won't include them — we force-add them.
        if user_context:
            def _parse_grams_from_str(qty: str):
                import re as _re
                if not qty:
                    return None
                qty = qty.strip().lower()
                # grams
                m = _re.match(r'(\d+(?:\.\d+)?)\s*g(?:rams?)?$', qty)
                if m:
                    return float(m.group(1))
                # ml (approximate 1ml = 1g)
                m = _re.match(r'(\d+(?:\.\d+)?)\s*ml$', qty)
                if m:
                    return float(m.group(1))
                # tablespoon / tbsp
                m = _re.match(r'(\d+(?:\.\d+)?)\s*(?:tablespoons?|tbsp)', qty)
                if m:
                    return float(m.group(1)) * 14.0
                # teaspoon / tsp
                m = _re.match(r'(\d+(?:\.\d+)?)\s*(?:teaspoons?|tsp)', qty)
                if m:
                    return float(m.group(1)) * 5.0
                # cup
                m = _re.match(r'(\d+(?:\.\d+)?)\s*cups?', qty)
                if m:
                    return float(m.group(1)) * 240.0
                # ounce / oz
                m = _re.match(r'(\d+(?:\.\d+)?)\s*(?:ounces?|oz)', qty)
                if m:
                    return float(m.group(1)) * 28.35
                # kg
                m = _re.match(r'(\d+(?:\.\d+)?)\s*kg', qty)
                if m:
                    return float(m.group(1)) * 1000.0
                return None

            # Collect all items to inject
            items_to_inject = []
            for item in user_context.get('hidden_ingredients', []):
                name = (item.get('name') or '').strip()
                if name:
                    items_to_inject.append({'name': name, 'quantity': item.get('quantity', ''), 'type': 'hidden'})
            for item in user_context.get('extras', []):
                name = (item.get('name') or '').strip()
                if name:
                    items_to_inject.append({'name': name, 'quantity': item.get('quantity', ''), 'type': 'extra'})

            # Ask Gemini to estimate calories for all injected items in one call
            gemini_kcal_map = {}
            if items_to_inject and self.config.GEMINI_API_KEY:
                try:
                    items_desc = ', '.join(
                        f"{i['name']} ({i['quantity']})" if i.get('quantity') else i['name']
                        for i in items_to_inject
                    )
                    cal_prompt = (
                        f"For each of these food items at the given quantities: {items_desc}. "
                        "Convert the quantity to grams and estimate the calories. "
                        "Return ONLY a JSON array like: "
                        "[{\"name\": \"butter\", \"grams\": 14.2, \"kcal\": 102}, ...]. No extra text."
                    )
                    cal_response = genai.GenerativeModel("gemini-2.0-flash").generate_content(cal_prompt)
                    cal_text = (cal_response.text or "").strip()
                    if "```" in cal_text:
                        cal_text = cal_text.split("```")[1]
                        if cal_text.startswith("json"):
                            cal_text = cal_text[4:]
                    cal_data = json.loads(cal_text.strip())
                    for entry in cal_data:
                        n = (entry.get('name') or '').strip().lower()
                        k = entry.get('kcal')
                        g = entry.get('grams')
                        if n and k is not None:
                            gemini_kcal_map[n] = float(k)
                        if n and g is not None:
                            # Only use Gemini's gram estimate if local parsing returned None
                            if n not in gemini_kcal_map or True:
                                gemini_kcal_map[f"{n}__grams"] = float(g)
                    logger.info(f"[Gemini detection] Calorie estimates for injected items: {gemini_kcal_map}")
                except Exception as e:
                    logger.warning(f"[Gemini detection] Could not estimate calories for injected items: {e}")

            for item in items_to_inject:
                name = item['name']
                g = _parse_grams_from_str(item.get('quantity', ''))
                # Fall back to Gemini's gram estimate if local parsing couldn't determine grams
                if g is None:
                    g = gemini_kcal_map.get(f"{name.lower()}__grams")
                kcal = gemini_kcal_map.get(name.lower())
                boxes.append([0.0, 0.0, 1.0, 1.0])  # placeholder bbox (bypasses SAM2 via tiny area)
                labels.append(name)
                grams_list.append(g)
                quantity_list.append(1)
                calories_list.append(kcal)
                logger.info(f"[Gemini detection] Injected {item['type']}: {name} ({g}g, {kcal} kcal)")

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.array([])
        print(f"  ✓ Gemini detection: {len(labels)} objects")
        sys.stdout.flush()
        return boxes, labels, caption, [], grams_list, quantity_list, calories_list
    
    def _detect_objects_gemini_multi_image(self, frames_list: List[np.ndarray], job_id: str, user_context: dict = None):
        """
        Detect food objects using Gemini with multiple images (5 frames) in one prompt.
        Same prompt logic as single image; additionally instructs: do not count duplicates
        across frames — list each unique food item once. Boxes are in first image coordinates.
        Returns (boxes, labels, caption, grams_list, quantity_list, ref_size) with ref_size=(w, h) of first frame.
        """
        import sys
        sys.stdout.flush()
        if not self.config.GEMINI_API_KEY:
            logger.warning("[Gemini multi-image] GEMINI_API_KEY not set")
            return None
        n_expected = getattr(self.config, "VIDEO_NUM_FRAMES", 5)
        if len(frames_list) < n_expected:
            logger.warning(f"[Gemini multi-image] Expected at least {n_expected} frames, got {len(frames_list)}")
            return None
        # Use first frame dimensions for bbox coordinates
        h0, w0 = frames_list[0].shape[:2]
        img_width, img_height = w0, h0
        # Same prompt as single image, plus: 5 frames, do not count duplicates
        prompt = (
            "These 5 images are consecutive frames (1 second apart) from a single 5-second video clip. "
            "Analyze them together. List each UNIQUE food item exactly once — do NOT count the same physical item "
            "twice if it appears in multiple frames. For each item provide bounding_box in the coordinate system "
            "of the FIRST image only.\n\n"
            "Analyze this food image in detail. Provide a comprehensive analysis including:\n"
            "1. MAIN DISH/FOOD ITEM: Primary food name, cuisine type, cooking method.\n"
            "2. VISIBLE INGREDIENTS WITH LOCATIONS: List all visible ingredients/components (garnishes, sides, sauces). "
            "For each visible food item provide bounding box [x_min, y_min, x_max, y_max], estimated_quantity_grams, and quantity (count).\n"
            f"Image dimensions (first image): {img_width} x {img_height} pixels. Bounding boxes in pixels (0 to width/height).\n"
            "estimated_quantity_grams: TOTAL edible mass in grams for that item. When there are multiple identical pieces (e.g. 6 kiwi slices, grapes, almonds), use ONE entry with quantity set to the count and estimated_quantity_grams as the TOTAL mass for all of them.\n"
            "quantity: For small countable items (nuts, almonds, berries, grapes, etc.) COUNT EVERY visible piece and set quantity to that exact total (e.g. 15 almonds -> quantity 15). Do not underestimate.\n"
            "Use realistic typical weights (e.g. one fish fillet 80–120g, 15 almonds ~15g total, six kiwi slices ~120g). "
            "Different ingredients should have different estimated_quantity_grams; for the same ingredient with many pieces use one entry and total mass.\n"
            "3. INGREDIENT BREAKDOWN, 4. NUTRITIONAL INFORMATION, 5. ADDITIONAL NOTES.\n\n"
            "Format as JSON: main_food_item, cuisine_type, cooking_method, "
            "visible_ingredients (array of {name, bounding_box [x_min,y_min,x_max,y_max], estimated_quantity_grams, quantity}), "
            "ingredient_breakdown, nutritional_info, allergens, dietary_tags, additional_notes.\n"
            "Example: [{\"name\": \"fish fillet\", \"bounding_box\": [100,50,300,250], \"estimated_quantity_grams\": 95, \"quantity\": 1}, {\"name\": \"almonds\", \"bounding_box\": [50,400,280,520], \"estimated_quantity_grams\": 15, \"quantity\": 15}]. "
            "Output only valid JSON (you may wrap in ```json)."
        )
        prompt += self._build_user_context_suffix(user_context)
        try:
            from google import genai as genai_new
            from google.genai import types
            client = genai_new.Client(api_key=self.config.GEMINI_API_KEY)
            parts = [types.Part(text=prompt)]
            for i, frame in enumerate(frames_list[:n_expected]):
                pil_img = Image.fromarray(frame)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                parts.append(types.Part(inline_data=types.Blob(data=buf.getvalue(), mime_type="image/jpeg")))
            response_text = ""
            for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
                try:
                    print(f"  → Calling Gemini multi-image for food detection ({model_name}), 5 frames (no duplicates)...")
                    sys.stdout.flush()
                    response = client.models.generate_content(
                        model=model_name,
                        contents=types.Content(parts=parts),
                    )
                    response_text = (response.text or "").strip()
                    if response_text:
                        break
                except Exception as e:
                    logger.warning(f"[Gemini multi-image] {model_name} failed: {e}")
                    continue
            if not response_text:
                logger.warning("[Gemini multi-image] All models failed")
                return None
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end] if json_start >= 0 else ""
            if not json_str:
                return None
            data = json.loads(json_str)
        except Exception as e:
            logger.warning(f"[Gemini multi-image] Failed: {e}")
            return None
        visible = data.get("visible_ingredients") or []
        boxes = []
        labels = []
        grams_list = []
        quantity_list = []
        for ing in visible:
            bbox = ing.get("bounding_box")
            name = (ing.get("name") or "").strip()
            if not name or not bbox or len(bbox) != 4:
                continue
            x_min = max(0, min(float(bbox[0]), img_width))
            y_min = max(0, min(float(bbox[1]), img_height))
            x_max = max(0, min(float(bbox[2]), img_width))
            y_max = max(0, min(float(bbox[3]), img_height))
            if x_max <= x_min or y_max <= y_min:
                continue
            g = ing.get("estimated_quantity_grams")
            try:
                grams_list.append(float(g) if g is not None else None)
            except (TypeError, ValueError):
                grams_list.append(None)
            q = ing.get("quantity")
            try:
                quantity_list.append(max(1, int(q)) if q is not None else 1)
            except (TypeError, ValueError):
                quantity_list.append(1)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(name)
        caption = data.get("main_food_item") or ""
        if data.get("additional_notes"):
            caption = f"{caption}. {data['additional_notes']}" if caption else data["additional_notes"]
        if not boxes and (data.get("main_food_item") or data.get("ingredient_breakdown")):
            fallback_labels = []
            main = (data.get("main_food_item") or "").strip()
            if main:
                fallback_labels.append(main)
            for x in (data.get("ingredient_breakdown") or []):
                if isinstance(x, str) and x.strip():
                    fallback_labels.append(x.strip())
                elif isinstance(x, dict) and (x.get("name") or x.get("item")):
                    fallback_labels.append((x.get("name") or x.get("item") or "").strip())
            if fallback_labels:
                seen = set()
                unique = [x for x in fallback_labels if x and x.lower() not in seen and not seen.add(x.lower())]
                boxes = [[0, 0, img_width, img_height]] * len(unique)
                labels = unique
                grams_list = [None] * len(unique)
                quantity_list = [1] * len(unique)
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.array([])
        print(f"  ✓ Gemini multi-image: {len(labels)} unique objects (no duplicates across frames)")
        sys.stdout.flush()
        ref_size = (img_width, img_height)
        return (boxes, labels, caption, grams_list, quantity_list, ref_size)
    
    # Reference resolution for Gemini video bounding boxes (prompt asks for 1280x720)
    _GEMINI_VIDEO_REF_W = 1280
    _GEMINI_VIDEO_REF_H = 720
    _GEMINI_VIDEO_INLINE_LIMIT = 20 * 1024 * 1024  # 20 MB
    
    def _detect_objects_gemini_video(self, video_path: Path, job_id: str, user_context: dict = None):
        """
        One-shot Gemini video understanding: call Gemini video API once for the whole clip.
        Returns (boxes, labels, caption) with boxes in reference resolution 1280x720
        so the pipeline can scale them to the actual frame size.
        """
        import sys
        sys.stdout.flush()
        if not self.config.GEMINI_API_KEY:
            logger.warning("[Gemini video] GEMINI_API_KEY not set")
            return None
        video_path = Path(video_path)
        if not video_path.exists():
            logger.warning(f"[Gemini video] File not found: {video_path}")
            return None
        prompt = (
            "Analyze this video from a food and nutrition perspective. Describe what is shown: meals, dishes, ingredients.\n"
            "Format the response as structured JSON with: main_food_item, cuisine_type, cooking_method, "
            "visible_ingredients (list of objects with: name, bounding_box [x_min, y_min, x_max, y_max] for a representative frame at 1280x720, estimated_quantity_grams, quantity, timestamp_seconds or time_range), "
            "ingredient_breakdown, nutritional_info, allergens, dietary_tags, additional_notes.\n"
            "Bounding boxes: [x_min, y_min, x_max, y_max] in pixels; frame size 1280x720.\n"
            "estimated_quantity_grams: TOTAL edible mass in grams for that item. When there are multiple identical pieces (e.g. 6 kiwi slices), use ONE entry with quantity set to the count and estimated_quantity_grams as the TOTAL mass for all.\n"
            "quantity: integer count of identical items (e.g. 6 for six kiwi slices, 1 for a single portion). Always include; use 1 for a single item.\n"
            "Use realistic typical weights (e.g. one fish fillet 80–120g, sauce 40–80g). Each item must have a DIFFERENT value reflecting its apparent portion size.\n"
            "Example: [{\"name\": \"fish fillet\", \"bounding_box\": [320,200,600,500], \"estimated_quantity_grams\": 100, \"quantity\": 1, \"timestamp_seconds\": 0}, {\"name\": \"Kiwi Slices\", \"bounding_box\": [100,400,400,600], \"estimated_quantity_grams\": 120, \"quantity\": 6}].\n"
            "Output only valid JSON (you may wrap in ```json)."
        )
        prompt += self._build_user_context_suffix(user_context)
        try:
            try:
                from google import genai as genai_new
                from google.genai import types
                client = genai_new.Client(api_key=self.config.GEMINI_API_KEY)
                size = video_path.stat().st_size
                mime = "video/mp4" if video_path.suffix.lower() in (".mp4", ".mpg", ".mpeg") else "video/quicktime"
                video_models_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
                response_text = ""
                if size <= self._GEMINI_VIDEO_INLINE_LIMIT:
                    video_bytes = video_path.read_bytes()
                    parts = [
                        types.Part(inline_data=types.Blob(data=video_bytes, mime_type=mime)),
                        types.Part(text=prompt),
                    ]
                    for model_name in video_models_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=types.Content(parts=parts),
                            )
                            response_text = response.text or ""
                            if response_text:
                                break
                        except Exception as model_err:
                            logger.warning(f"[Gemini video] Model {model_name} error: {model_err}")
                            continue
                else:
                    print("  → Uploading video via File API (Gemini video)...")
                    sys.stdout.flush()
                    myfile = client.files.upload(file=str(video_path))
                    for model_name in video_models_try:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[myfile, prompt],
                            )
                            response_text = response.text or ""
                            if response_text:
                                break
                        except Exception as model_err:
                            logger.warning(f"[Gemini video] Model {model_name} error: {model_err}")
                            continue
            except ImportError as ie:
                logger.warning(f"[Gemini video] Import error: {ie}")
                response_text = ""
            if not response_text:
                print("  [Gemini video] No response text from API")
                sys.stdout.flush()
                return None
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end] if json_start >= 0 else ""
            if not json_str:
                print("  [Gemini video] No JSON found in response")
                sys.stdout.flush()
                return None
            data = json.loads(json_str)
        except Exception as e:
            logger.warning(f"[Gemini video] Failed: {e}")
            print(f"  [Gemini video] Exception: {e}")
            sys.stdout.flush()
            return None
        visible = data.get("visible_ingredients") or []
        boxes = []
        labels = []
        grams_list = []
        quantity_list = []
        ref_w, ref_h = self._GEMINI_VIDEO_REF_W, self._GEMINI_VIDEO_REF_H
        for ing in visible:
            bbox = ing.get("bounding_box")
            name = (ing.get("name") or "").strip()
            if not name or not bbox or len(bbox) != 4:
                continue
            x_min = max(0, min(float(bbox[0]), ref_w))
            y_min = max(0, min(float(bbox[1]), ref_h))
            x_max = max(0, min(float(bbox[2]), ref_w))
            y_max = max(0, min(float(bbox[3]), ref_h))
            if x_max <= x_min or y_max <= y_min:
                continue
            g = ing.get("estimated_quantity_grams")
            try:
                grams_list.append(float(g) if g is not None else None)
            except (TypeError, ValueError):
                grams_list.append(None)
            q = ing.get("quantity")
            try:
                quantity_list.append(max(1, int(q)) if q is not None else 1)
            except (TypeError, ValueError):
                quantity_list.append(1)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(name)
        caption = data.get("main_food_item") or ""
        if data.get("additional_notes"):
            caption = f"{caption}. {data['additional_notes']}" if caption else data["additional_notes"]
        if not boxes:
            # Fallback: use main_food_item or ingredient_breakdown as labels with full-frame box so nutrition runs
            fallback_labels = []
            main = (data.get("main_food_item") or "").strip()
            if main:
                fallback_labels.append(main)
            breakdown = data.get("ingredient_breakdown")
            if isinstance(breakdown, list):
                for x in breakdown:
                    if isinstance(x, str) and x.strip():
                        fallback_labels.append(x.strip())
                    elif isinstance(x, dict) and (x.get("name") or x.get("item")):
                        fallback_labels.append((x.get("name") or x.get("item") or "").strip())
            elif isinstance(breakdown, str) and breakdown.strip():
                for part in breakdown.replace(",", "\n").split():
                    if part.strip():
                        fallback_labels.append(part.strip())
            if fallback_labels:
                # Dedupe preserving order
                seen = set()
                unique = [x for x in fallback_labels if x and x.lower() not in seen and not seen.add(x.lower())]
                boxes = [[0, 0, ref_w, ref_h]] * len(unique)
                labels = unique
                grams_list = [None] * len(unique)
                quantity_list = [1] * len(unique)
                print(f"  [Gemini video] No bounding boxes; using {len(labels)} items from description (full-frame)")
                sys.stdout.flush()
            else:
                print("  [Gemini video] No visible_ingredients with valid bbox and no fallback text")
                sys.stdout.flush()
                return None
        print(f"  ✓ Gemini video (one-shot): {len(labels)} objects")
        sys.stdout.flush()
        return (np.array(boxes, dtype=np.float32), labels, caption, grams_list, quantity_list, (self._GEMINI_VIDEO_REF_W, self._GEMINI_VIDEO_REF_H))
    
    def _detect_objects_florence(self, image_pil, processor, model):
        """Detect objects using Florence-2 (used when USE_GEMINI_DETECTION is False)."""
        import sys
        sys.stdout.flush()
        
        # Track ingredients that VQA identifies but grounding can't localize
        unquantified_ingredients = []
        
        # Check if using hybrid detection (combines OD + detailed caption)
        if self.config.caption_type == "hybrid_detection":
            print("  → Using hybrid detection (OD + detailed caption)...")
            sys.stdout.flush()
            
            # Method 1: Direct object detection
            print("    → Step 1: Direct object detection (OD)...")
            sys.stdout.flush()
            od_results = self._run_florence2("<OD>", None, image_pil, processor, model)
            od_data = od_results.get("<OD>", {})
            od_boxes = np.array(od_data.get("bboxes", []))
            od_labels = od_data.get("labels", [])
            print(f"    ✓ OD found {len(od_boxes)} objects")
            sys.stdout.flush()
            
            # Method 2: Detailed caption + grounding
            print("    → Step 2: Detailed caption + phrase grounding...")
            sys.stdout.flush()
            caption_results = self._run_florence2("<MORE_DETAILED_CAPTION>", None, image_pil, processor, model)
            caption = caption_results["<MORE_DETAILED_CAPTION>"]
            print(f"    ✓ Caption: {caption[:100]}...")
            sys.stdout.flush()
            
            grounding_results = self._run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', caption, image_pil, processor, model)
            grounding_data = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
            caption_boxes = np.array(grounding_data.get("bboxes", []))
            caption_labels = grounding_data.get("labels", [])
            print(f"    ✓ Caption grounding found {len(caption_boxes)} objects")
            sys.stdout.flush()
            
            # Merge results: Use OD detections as primary source (most accurate, no hallucinations)
            # Only add caption detections for objects that OD didn't detect
            print("    → Merging detection results (OD-first approach)...")
            sys.stdout.flush()
            all_boxes = []
            all_labels = []
            
            # Add all OD detections first (these are grounded and accurate)
            for od_box, od_label in zip(od_boxes, od_labels):
                all_boxes.append(od_box)
                all_labels.append(od_label)
            
            # Only add caption detections that don't overlap with OD detections
            # This avoids hallucinations from caption generation
            for cap_box, cap_label in zip(caption_boxes, caption_labels):
                # Check if this caption detection overlaps significantly with any OD detection
                overlaps_with_od = False
                for od_box in od_boxes:
                    iou = self._calculate_iou(cap_box, od_box)
                    if iou > 0.3:  # If >30% overlap, OD already detected it - skip caption (avoids hallucinations)
                        overlaps_with_od = True
                        break
                
                if not overlaps_with_od:
                    # Only add if OD didn't detect this object
                    all_boxes.append(cap_box)
                    all_labels.append(cap_label)
            
            # Deduplicate: remove overlapping detections with similar labels
            boxes, labels = self._deduplicate_detections(all_boxes, all_labels)
            
            # Store full caption (don't truncate)
            caption = f"Hybrid detection: {caption}"
            print(f"  ✓ Hybrid detection complete: {len(boxes)} total objects (OD: {len(od_boxes)}, Caption: {len(caption_boxes)}, After dedup: {len(boxes)})")
            sys.stdout.flush()
            
        elif self.config.caption_type == "detailed_od":
            # Detailed OD: Use OD for accuracy, enhance with basic caption for more specific labels
            print("  → Using detailed OD (OD + basic caption for enhanced labels)...")
            sys.stdout.flush()
            
            # Step 1: Direct object detection (accurate, no hallucinations)
            print("    → Step 1: Direct object detection (OD)...")
            sys.stdout.flush()
            od_results = self._run_florence2("<OD>", None, image_pil, processor, model)
            od_data = od_results.get("<OD>", {})
            od_boxes = np.array(od_data.get("bboxes", []))
            od_labels = od_data.get("labels", [])
            print(f"    ✓ OD found {len(od_boxes)} objects")
            sys.stdout.flush()
            
            # Step 2: Basic caption (less prone to hallucinations than MORE_DETAILED_CAPTION)
            print("    → Step 2: Basic caption for context...")
            sys.stdout.flush()
            caption_results = self._run_florence2("<DETAILED_CAPTION>", None, image_pil, processor, model)
            caption = caption_results.get("<DETAILED_CAPTION>", "")
            print(f"    ✓ Caption: {caption[:100]}...")
            sys.stdout.flush()
            
            # Step 3: Ground caption phrases to get more specific labels
            print("    → Step 3: Grounding caption phrases...")
            sys.stdout.flush()
            grounding_results = self._run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', caption, image_pil, processor, model)
            grounding_data = grounding_results.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
            caption_boxes = np.array(grounding_data.get("bboxes", []))
            caption_labels = grounding_data.get("labels", [])
            print(f"    ✓ Caption grounding found {len(caption_boxes)} phrases")
            sys.stdout.flush()
            
            # Step 4: Enhance OD labels with caption labels when they match
            print("    → Step 4: Enhancing OD labels with caption details...")
            sys.stdout.flush()
            enhanced_labels = []
            enhanced_boxes = []
            
            # For each OD detection, try to find a matching caption label that's more specific
            for od_box, od_label in zip(od_boxes, od_labels):
                best_match_label = od_label
                best_iou = 0
                
                # Find caption label that overlaps with this OD box and is more specific
                for cap_box, cap_label in zip(caption_boxes, caption_labels):
                    iou = self._calculate_iou(od_box, cap_box)
                    if iou > 0.3 and iou > best_iou:
                        # Check if caption label is more specific (longer, more descriptive)
                        # But avoid if it contains color adjectives (hallucination risk)
                        cap_label_lower = cap_label.lower()
                        has_color_hallucination = any(color in cap_label_lower.split() for color in 
                                                     ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink'])
                        
                        if not has_color_hallucination and len(cap_label) > len(od_label):
                            best_match_label = cap_label
                            best_iou = iou
                
                enhanced_boxes.append(od_box)
                enhanced_labels.append(best_match_label)
            
            # Add caption detections that don't overlap with OD (new objects)
            for cap_box, cap_label in zip(caption_boxes, caption_labels):
                overlaps_with_od = False
                for od_box in od_boxes:
                    iou = self._calculate_iou(cap_box, od_box)
                    if iou > 0.3:
                        overlaps_with_od = True
                        break
                
                if not overlaps_with_od:
                    # Check for color hallucinations before adding
                    cap_label_lower = cap_label.lower()
                    has_color_hallucination = any(color in cap_label_lower.split() for color in 
                                                 ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink'])
                    if not has_color_hallucination:
                        enhanced_boxes.append(cap_box)
                        enhanced_labels.append(cap_label)
            
            boxes = np.array(enhanced_boxes)
            labels = enhanced_labels
            
            # Deduplicate
            boxes, labels = self._deduplicate_detections(boxes, labels)
            
            caption = f"Detailed OD: {caption}"
            print(f"  ✓ Detailed OD complete: {len(boxes)} objects (enhanced from {len(od_boxes)} OD detections)")
            sys.stdout.flush()
            
        elif self.config.caption_type == "object_detection":
            print("  → Detecting objects directly (OD task)...")
            sys.stdout.flush()
            od_results = self._run_florence2("<OD>", None, image_pil, processor, model)
            od_data = od_results.get("<OD>", {})
            
            boxes = np.array(od_data.get("bboxes", []))
            labels = od_data.get("labels", [])
            caption = f"Detected objects: {', '.join(labels) if len(labels) > 0 else 'none'}"
            print(f"  ✓ Object detection complete: found {len(boxes)} objects")
            sys.stdout.flush()
            
        elif self.config.caption_type == "vqa":
            # VQA mode: Ask food-focused questions to get food items
            # Format: <VQA> + question (task token followed by question)
            print("  → Using VQA (Visual Question Answering) for food detection...")
            sys.stdout.flush()
            
            # Ask food-focused questions
            food_items = []
            for question in self.config.VQA_QUESTIONS:
                print(f"    → Asking: {question}")
                sys.stdout.flush()
                try:
                    # VQA: Use <VQA> task token followed by the question
                    vqa_result = self._run_florence2("<VQA>", question, image_pil, processor, model)
                    
                    # Debug: Log raw result for troubleshooting
                    logger.debug(f"Raw VQA result for question '{question}': {vqa_result}")
                    
                    # Extract answer from VQA result
                    answer = vqa_result.get("<VQA>", "")
                    if isinstance(answer, dict):
                        # If it's a dict, try to get text/answer field
                        answer = answer.get("answer", answer.get("text", str(answer)))
                    elif not isinstance(answer, str):
                        answer = str(answer)
                    
                    # Debug: Log raw answer before cleaning
                    logger.debug(f"Raw answer before cleaning: '{answer}'")
                    print(f"    → Raw answer: {answer[:200]}...")  # Show raw answer for debugging
                    sys.stdout.flush()
                    
                    # Use Gemini to format the answer (already configured for RAG fallback)
                    # Gemini understands semantic meaning, avoiding hardcoded patterns
                    if self.config.GEMINI_API_KEY:
                        try:
                            print(f"    → Formatting with Gemini...")
                            sys.stdout.flush()
                            
                            import google.generativeai as genai
                            import time
                            time.sleep(0.2)  # Rate limiting
                            genai.configure(api_key=self.config.GEMINI_API_KEY)
                            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
                            
                            prompt = (
                                f"Extract only the food item names from this text and list them separated by commas. "
                                f"Do not include utensils, plates, tables, or locations. "
                                f"Keep multi-word food names together (e.g., 'chicken nuggets', 'ice tea'). "
                                f"Text: {answer} "
                                f"Answer with just the comma-separated list:"
                            )
                            
                            response = gemini_model.generate_content(prompt)
                            formatted_answer = response.text.strip()
                            print(f"    → Gemini output: {formatted_answer[:100]}...")
                            sys.stdout.flush()
                            answer = formatted_answer
                        except Exception as e:
                            logger.warning(f"Gemini formatting failed: {e}, using simple text normalization")
                            # Fallback to simple normalization
                            answer = re.sub(r'\s+and\s+', ', ', answer, flags=re.IGNORECASE)
                            answer = re.sub(r'\s+on\s+(a\s+)?(wooden|white|blue)\s+(table|plate|board)', '', answer, flags=re.IGNORECASE)
                            print(f"    → Cleaned answer (fallback): {answer[:100]}...")
                            sys.stdout.flush()
                    else:
                        # No Gemini key - use simple text normalization
                        answer = re.sub(r'\s+and\s+', ', ', answer, flags=re.IGNORECASE)
                        answer = re.sub(r'\s+on\s+(a\s+)?(wooden|white|blue)\s+(table|plate|board)', '', answer, flags=re.IGNORECASE)
                        print(f"    → Cleaned answer: {answer[:100]}...")
                        sys.stdout.flush()
                    
                    answer = answer.strip()
                    
                    # Check if answer contains structured/XML-like tags (poly, loc, etc.) - this indicates grounding output, not VQA text
                    # If so, extract only the text portion before any XML tags appear
                    xml_tag_pattern = r'<[^>]+>'
                    if re.search(xml_tag_pattern, answer):
                        # Extract text before first XML tag
                        text_before_xml = re.split(xml_tag_pattern, answer)[0].strip()
                        if text_before_xml and len(text_before_xml) > 5:
                            answer = text_before_xml
                            print(f"    ⚠ Detected structured output, extracted text portion: {answer[:100]}...")
                            sys.stdout.flush()
                        else:
                            # If no meaningful text before XML, skip this answer
                            print(f"    ⚠ Skipping structured output (grounding format, not VQA text): {answer[:100]}...")
                            sys.stdout.flush()
                            continue
                    
                    # Remove task tokens and special tokens
                    answer = answer.replace("<VQA>", "").replace("<|endoftext|>", "").strip()
                    
                    # Check if answer is just repeating the question (common Florence-2 issue)
                    answer_lower = answer.lower()
                    question_lower = question.lower()
                    
                    # If answer has conversational fluff, try to extract the actual food list
                    # Pattern 1: "X? yes, ..." or "X. yes, ..." -> keep just X
                    # Pattern 2: "... are X, Y, Z" -> extract X, Y, Z
                    # Pattern 3: "X, Y, and Z are visible" -> keep X, Y, and Z
                    
                    # Remove conversational confirmations: "yes,", "sure,", etc.
                    answer = re.sub(r'\.\s*(yes|sure|ok|okay|right|correct)[,\s]+.*$', '', answer, flags=re.IGNORECASE)
                    answer = re.sub(r'\?\s*(yes|sure|ok|okay|right|correct)[,\s]+.*$', '', answer, flags=re.IGNORECASE)
                    
                    # Extract content after separators if it looks like a list
                    list_separators = [
                        (r'^.*?\s+are\s+', ''),  # "... are X, Y, Z" -> "X, Y, Z"
                        (r'^.*?\s+is\s+', ''),   # "... is X, Y, Z" -> "X, Y, Z"
                        (r'^.*?:\s*', ''),       # "...: X, Y, Z" -> "X, Y, Z"
                    ]
                    
                    original_answer = answer
                    for pattern, replacement in list_separators:
                        if re.search(pattern, answer, flags=re.IGNORECASE):
                            extracted = re.sub(pattern, replacement, answer, count=1, flags=re.IGNORECASE).strip()
                            # Only use extraction if result has commas or is short (looks like a list)
                            if ',' in extracted or len(extracted.split()) <= 10:
                                answer = extracted
                                break
                    
                    if answer != original_answer:
                        print(f"    → Extracted list from answer: {answer[:80]}...")
                        sys.stdout.flush()
                    
                    # Clean answer in stages for better results
                    
                    # Stage 1: Normalize "and" to commas FIRST (before spatial cleaning)
                    # This prevents "and" from being lost during spatial cleaning
                    answer = re.sub(r'\s+and\s+', ', ', answer, flags=re.IGNORECASE)
                    
                    # Stage 2: Remove spatial/descriptive phrases
                    # These add location/arrangement info but aren't food names
                    spatial_phrases = [
                        r'\s+on\s+top\s+of',           # "on top of" -> ""
                        r'\s+on\s+(the\s+)?(left|right|top|bottom|center|middle|table|board|plate)',
                        r'\s+in\s+(the\s+)?(bowl|plate|dish|container|background|foreground)',
                        r'\s+at\s+(the\s+)?(left|right|top|bottom|center|middle)',
                        r'\s+with\s+(a\s+)?(cutting\s+board|wooden\s+table)',
                    ]
                    
                    for pattern in spatial_phrases:
                        answer = re.sub(pattern, ',', answer, flags=re.IGNORECASE)
                    
                    # Stage 3: Clean up leftover prepositions that create malformed names
                    # "parmesan cheese of blue sauce" -> "parmesan cheese, blue sauce"
                    answer = re.sub(r'\s+(of|with|from)\s+', ', ', answer, flags=re.IGNORECASE)
                    
                    # Stage 4: Clean up multiple commas and extra spaces
                    answer = re.sub(r',\s*,+', ',', answer)  # ",," -> ","
                    answer = re.sub(r'\s+', ' ', answer)     # Multiple spaces -> one space
                    answer = answer.strip(', ')
                    
                    print(f"    → Cleaned answer: {answer[:80]}...")
                    sys.stdout.flush()
                    
                    # Final check: if answer has no content words left after cleaning, skip it
                    if len(answer.strip()) < 3:
                        print(f"    ⚠ Skipping empty answer after cleaning")
                        sys.stdout.flush()
                        continue
                    
                    # Now validate the extracted/cleaned answer
                    # If it's still mostly question words, skip it
                    question_keywords = set([w for w in question_lower.split() if len(w) > 3])
                    answer_keywords = set([w for w in answer.lower().split() if len(w) > 3])
                    if len(question_keywords) > 0 and len(answer_keywords) > 0:
                        # Check what % of answer words are question words (not the other way around)
                        overlap_ratio = len(question_keywords.intersection(answer_keywords)) / len(answer_keywords)
                        if overlap_ratio > 0.5:  # More than 50% of answer is question words
                            print(f"    ⚠ Skipping answer that is mostly question words ({overlap_ratio:.1%})")
                            sys.stdout.flush()
                            continue
                    
                    # Remove task tokens
                    answer = answer.replace("<VQA>", "").replace("<|endoftext|>", "").strip()
                    answer_lower = answer.lower()
                    
                    # Remove "vqa" artifacts (case-insensitive)
                    answer = answer.replace("vqa", "").replace("VQA", "").replace("Vqa", "").replace("vQa", "").replace("vqA", "").strip()
                    # Remove "list" if it's at the start (from "List all the food items")
                    if answer.lower().startswith("list "):
                        answer = answer[5:].strip()
                    # Remove "all the food items" artifacts
                    answer = answer.replace("all the food items", "").replace("All the food items", "").strip()
                    # Remove standalone "list" word (artifact from "vQAList")
                    if answer.lower() == "list" or answer.lower().startswith("list "):
                        answer = answer[4:].strip() if len(answer) > 4 else ""
                    
                    # Only filter out completely empty or obviously invalid answers
                    answer_lower = answer.lower().strip()
                    if not answer_lower or answer_lower in ["", "vqa", "vqalist"]:
                        print(f"    ⚠ Skipping empty answer")
                        sys.stdout.flush()
                        continue
                    
                    # Accept any answer that has content (removed aggressive filtering)
                    if answer and len(answer.strip()) > 1:
                        print(f"    ✓ Answer: {answer[:100]}...")
                        sys.stdout.flush()
                        food_items.append(answer)
                except Exception as e:
                    logger.warning(f"VQA question '{question}' failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Combine answers and extract food items with confidence scoring
            if food_items:
                # Count mentions across multiple VQA questions for confidence
                from collections import Counter
                food_mention_counts = Counter()  # Track how many times each food is mentioned
                
                for item in food_items:
                    # FLAN-T5 already formatted this as comma-separated
                    # Simple split by comma - no hardcoded patterns needed
                    parts = [p.strip() for p in item.split(",") if p.strip()]
                    
                    # Process each part - keep items as extracted without aggressive expansion
                    expanded_parts = []
                    for part in parts:
                        part = part.strip(".,;:!?").strip()
                        if not part:
                            continue
                        
                        # Use items as-is - trust Florence's natural word boundaries
                        # Grounding will filter out any that don't exist in the image
                        expanded_parts.append(part)
                    
                    for part in expanded_parts:
                        part = part.strip()
                        if not part or len(part) < 3:
                            continue
                        
                        # Basic validation: should contain at least one letter
                        if not any(c.isalpha() for c in part):
                            continue
                        
                        # Filter non-food items (containers, surfaces, utensils)
                        part_lower = part.lower()
                        non_food_keywords = [
                            'board', 'cutting', 'wooden', 'plate', 'bowl', 'dish', 
                            'table', 'surface', 'counter', 'tray', 'platter',
                            'napkin', 'fork', 'knife', 'spoon', 'glass', 'cup'
                        ]
                        if any(keyword in part_lower for keyword in non_food_keywords):
                            # Skip unless it's a compound food name like "cheese board"
                            if not any(food_word in part_lower for food_word in ['cheese', 'charcuterie']):
                                continue
                        
                        # Normalize to lowercase for counting
                        food_name_lower = part.lower().strip()
                        if len(food_name_lower) < 3:
                            continue
                        
                        # Count this mention
                        food_mention_counts[food_name_lower] += 1
                
                # Convert to final list
                all_food_mentions = []
                for food_lower, count in food_mention_counts.items():
                    # Title case for display
                    food_name = food_lower.title()
                    all_food_mentions.append(food_name)
                
                # Remove exact duplicates while preserving order
                seen = set()
                unique_foods = []
                for food in all_food_mentions:
                    food_lower = food.lower()
                    if food_lower not in seen:
                        seen.add(food_lower)
                        unique_foods.append(food)
                
                if unique_foods:
                    # Use VQA extracted food items directly - grounding only to get bounding boxes
                    combined_answer = ", ".join(unique_foods)
                    print(f"  → Extracted food items: {combined_answer}")
                    print(f"  → Getting bounding boxes via grounding...")
                    sys.stdout.flush()
                    
                    # Run grounding to get bounding boxes for SAM2 segmentation
                    grounding_results = self._run_florence2(
                        '<CAPTION_TO_PHRASE_GROUNDING>', combined_answer, image_pil, processor, model
                    )
                    grounding_data = grounding_results.get('<CAPTION_TO_PHRASE_GROUNDING>', {})
                    boxes = np.array(grounding_data.get("bboxes", []))
                    labels = grounding_data.get("labels", [])
                    
                    # Track items that VQA identified but grounding couldn't locate
                    # These are ingredients mixed into dishes (e.g., cheese in pasta)
                    grounded_items_lower = [l.lower() for l in labels]
                    unquantified_ingredients = []
                    for food in unique_foods:
                        # Check if this food item got a bounding box
                        if not any(food.lower() in grounded.lower() or grounded.lower() in food.lower() 
                                   for grounded in grounded_items_lower):
                            unquantified_ingredients.append(food)
                    
                    if unquantified_ingredients:
                        print(f"  → Detected but not localized (mixed ingredients): {', '.join(unquantified_ingredients)}")
                        sys.stdout.flush()
                    
                    # Use grounding results directly - no complex filtering
                    # Grounding naturally filters out items that don't exist (they won't get boxes)
                    caption = f"VQA detection: {combined_answer}"
                    print(f"  ✓ VQA complete: found {len(boxes)} objects with bounding boxes")
                    sys.stdout.flush()
                else:
                    # Fallback if no food items extracted
                    logger.warning("No food items extracted from VQA answers, falling back to OD")
                    od_results = self._run_florence2("<OD>", None, image_pil, processor, model)
                    od_data = od_results.get("<OD>", {})
                    boxes = np.array(od_data.get("bboxes", []))
                    labels = od_data.get("labels", [])
                    caption = f"VQA fallback to OD: {', '.join(labels) if len(labels) > 0 else 'none'}"
            else:
                # Fallback to OD if VQA fails
                logger.warning("VQA failed, falling back to OD")
                od_results = self._run_florence2("<OD>", None, image_pil, processor, model)
                od_data = od_results.get("<OD>", {})
                boxes = np.array(od_data.get("bboxes", []))
                labels = od_data.get("labels", [])
                caption = f"VQA fallback to OD: {', '.join(labels) if len(labels) > 0 else 'none'}"
            
            # Filter color hallucinations
            def filter_color_hallucinations(label):
                colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
                words = label.lower().split()
                filtered_words = []
                for i, word in enumerate(words):
                    if word in colors:
                        if i + 1 < len(words):
                            next_word = words[i + 1]
                            food_compounds = ['pepper', 'bean', 'rice', 'tea', 'coffee', 'sauce', 'bread', 'cheese']
                            if next_word not in food_compounds:
                                continue
                        else:
                            continue
                    filtered_words.append(word)
                result = ' '.join(filtered_words).strip()
                return result.capitalize() if result else label
            
            labels = [filter_color_hallucinations(label) for label in labels]
            boxes, labels = self._deduplicate_detections(boxes, labels)
            
        else:
            # Generate caption (caption-based detection for detailed labels)
            print(f"  → Generating caption ({self.config.caption_type})...")
            sys.stdout.flush()
            caption_task = self.TASK_PROMPTS[self.config.caption_type]
            # Note: Florence-2 requires task token to be the only text, so we can't add custom prompts
            caption_results = self._run_florence2(caption_task, None, image_pil, processor, model)
            caption = caption_results[caption_task]
            print(f"  ✓ Caption: {caption[:150]}...")
            sys.stdout.flush()
            
            # Phrase grounding
            print("  → Grounding phrases to bounding boxes...")
            sys.stdout.flush()
            grounding_results = self._run_florence2(
                '<CAPTION_TO_PHRASE_GROUNDING>', caption, image_pil, processor, model
            )
            print("  ✓ Grounding complete")
            sys.stdout.flush()
            grounding_data = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
            
            boxes = np.array(grounding_data.get("bboxes", []))
            labels = grounding_data.get("labels", [])
            
            # Filter out color hallucinations from labels
            def filter_color_hallucinations(label):
                """Remove color adjectives that are likely hallucinations"""
                colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
                words = label.lower().split()
                filtered_words = []
                for i, word in enumerate(words):
                    if word in colors:
                        # Keep colors only if part of compound food names
                        if i + 1 < len(words):
                            next_word = words[i + 1]
                            food_compounds = ['pepper', 'bean', 'rice', 'tea', 'coffee', 'sauce', 'bread', 'cheese']
                            if next_word not in food_compounds:
                                continue  # Skip color word (likely hallucination)
                        else:
                            continue  # Skip standalone color words
                    filtered_words.append(word)
                result = ' '.join(filtered_words).strip()
                return result.capitalize() if result else label
            
            # Filter hallucinations from all labels
            labels = [filter_color_hallucinations(label) for label in labels]
            
            # Deduplicate caption-based detections (can have duplicates like "burgers" and "The burgers")
            boxes, labels = self._deduplicate_detections(boxes, labels)
        
        # Filter generic objects
        filtered_boxes = []
        filtered_labels = []
        for box, label in zip(boxes, labels):
            if label.lower() not in self.config.GENERIC_OBJECTS:
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                if box_area >= self.config.MIN_BOX_AREA:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
        
        return np.array(filtered_boxes), filtered_labels, caption, unquantified_ingredients
    
    def _is_likely_hallucination(self, food_name: str) -> bool:
        """
        Detect if a food name is likely a hallucination based on unlikely combinations.
        
        Args:
            food_name: The food item name to check
            
        Returns:
            True if likely a hallucination, False otherwise
        """
        food_lower = food_name.lower()
        
        # Unlikely food combinations that suggest hallucinations
        unlikely_patterns = [
            # Fruit + savory combinations that are very rare
            r'blueberry.*(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle)',
            r'(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle).*blueberry',
            r'strawberry.*(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle)',
            r'(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle).*strawberry',
            r'apple.*(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle)',
            r'(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle).*apple',
            
            # Ice cream + savory combinations
            r'ice cream.*(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle|fries)',
            r'(cheeseburger|burger|sandwich|pizza|pasta|rice|noodle|fries).*ice cream',
            
            # Multiple unlikely modifiers together
            r'blueberry.*blueberry',  # Repeated unlikely words
            r'blue.*blue',  # Repeated color words
            
            # Very long compound names (often hallucinations)
            r'^.{40,}$',  # Names longer than 40 chars are suspicious
            
            # Unlikely combinations of common foods
            r'(fries|french fries).*(ice tea|tea|coffee|soda)',
            r'(ice tea|tea|coffee|soda).*(fries|french fries)',
        ]
        
        # Check against patterns
        for pattern in unlikely_patterns:
            if re.search(pattern, food_lower):
                return True
        
        # Check for multiple color words (except in legitimate contexts)
        color_words = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
        color_count = sum(1 for word in food_lower.split() if word in color_words)
        if color_count > 1:
            # Multiple colors in one food name is suspicious
            return True
        
        # Check for very specific unlikely combinations
        unlikely_combos = [
            'blueberry cheeseburger',
            'blueberry ice cream sandwich',
            'blueberry french fries',
            'blueberry ice tea',
            'blue cheeseburger',
            'blue ice cream',
        ]
        
        for combo in unlikely_combos:
            if combo in food_lower:
                return True
        
        return False
    
    def _deduplicate_detections(self, boxes, labels):
        """
        Remove duplicate detections that overlap significantly or are very close.
        Uses IoU, center distance, and label similarity to detect duplicates.
        """
        if len(boxes) == 0:
            return np.array([]), []
        
        # Normalize labels for comparison (remove articles, lowercase, handle plurals)
        def normalize_label(label):
            label_lower = label.lower().strip()
            # Remove articles
            for article in ['a ', 'an ', 'the ']:
                if label_lower.startswith(article):
                    label_lower = label_lower[len(article):].strip()
            # Handle plurals (simple: remove trailing 's')
            if label_lower.endswith('s') and len(label_lower) > 1:
                # Don't remove 's' from words like "glass" -> "glas"
                if label_lower not in ['glass', 'glasses', 'fries', 'nuggets']:
                    label_lower = label_lower[:-1]
            return label_lower
        
        normalized_labels = [normalize_label(l) for l in labels]
        
        # Track which boxes to keep
        keep = [True] * len(boxes)
        
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            
            box_i = boxes[i]
            label_i = normalized_labels[i]
            
            # Calculate box center and size
            center_i = np.array([(box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2])
            size_i = np.array([box_i[2] - box_i[0], box_i[3] - box_i[1]])
            area_i = size_i[0] * size_i[1]
            
            for j in range(i + 1, len(boxes)):
                if not keep[j]:
                    continue
                
                box_j = boxes[j]
                label_j = normalized_labels[j]
                
                # Check if labels are similar
                labels_similar = (label_i == label_j or 
                                 label_i in label_j or 
                                 label_j in label_i)
                
                if not labels_similar:
                    continue
                
                # Check IoU
                iou = self._calculate_iou(box_i, box_j)
                
                # Calculate center distance and size similarity
                center_j = np.array([(box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2])
                size_j = np.array([box_j[2] - box_j[0], box_j[3] - box_j[1]])
                area_j = size_j[0] * size_j[1]
                
                center_dist = np.linalg.norm(center_i - center_j)
                avg_size = (size_i + size_j) / 2
                max_size = np.max(avg_size)
                
                # Size similarity (how similar are the box sizes)
                size_ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 0
                
                # Consider duplicate if:
                # 1. High IoU (>20%) with same label, OR
                # 2. Same label + centers are close (<50% of average box size) + similar sizes (>60% size ratio), OR
                # 3. One box contains the other (large box contains smaller box of same label)
                is_duplicate = False
                
                # Check if one box contains the other
                def box_contains(box_a, box_b):
                    """Check if box_a contains box_b"""
                    return (box_a[0] <= box_b[0] and box_a[1] <= box_b[1] and 
                           box_a[2] >= box_b[2] and box_a[3] >= box_b[3])
                
                if iou > 0.2:
                    is_duplicate = True
                elif center_dist < max_size * 0.5 and size_ratio > 0.6:
                    is_duplicate = True
                elif box_contains(box_i, box_j) or box_contains(box_j, box_i):
                    # If one box contains another with same label, keep the smaller one (more specific)
                    if area_i > area_j * 1.5:  # box_i is much larger
                        keep[i] = False  # Remove the large box
                        break
                    elif area_j > area_i * 1.5:  # box_j is much larger
                        keep[j] = False  # Remove the large box
                    else:
                        is_duplicate = True
                
                if is_duplicate:
                    # Keep the one with larger area (more complete detection) or more descriptive label
                    if area_i >= area_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        
        # Filter to keep only non-duplicates
        filtered_boxes = [boxes[i] for i in range(len(boxes)) if keep[i]]
        filtered_labels = [labels[i] for i in range(len(boxes)) if keep[i]]
        
        # Additional pass: Remove boxes that are too large (probably detecting groups)
        # Calculate average box area for each label
        if len(filtered_boxes) > 0:
            label_areas = {}
            for box, label in zip(filtered_boxes, filtered_labels):
                area = (box[2] - box[0]) * (box[3] - box[1])
                if label not in label_areas:
                    label_areas[label] = []
                label_areas[label].append(area)
            
            # Remove boxes that are much larger than average for their label
            final_boxes = []
            final_labels = []
            for i, (box, label) in enumerate(zip(filtered_boxes, filtered_labels)):
                area = (box[2] - box[0]) * (box[3] - box[1])
                avg_area = np.mean(label_areas[label])
                
                # If box is more than 3x larger than average, it's probably detecting a group
                if area > avg_area * 3 and len(label_areas[label]) > 1:
                    logger.info(f"Skipping oversized box for '{label}': area={area:.0f} vs avg={avg_area:.0f}")
                    continue
                
                final_boxes.append(box)
                final_labels.append(label)
            
            return np.array(final_boxes) if final_boxes else np.array([]), final_labels
        
        return np.array(filtered_boxes) if filtered_boxes else np.array([]), filtered_labels
    
    # Very distinct color palette (RGB 0-255) – maximally separated hues so items
    # are easy to tell apart even when many are on screen at once.
    DISTINCT_COLORS_RGB = [
        (230,  25,  75),   # Red
        ( 60, 180,  75),   # Green
        (  0, 130, 200),   # Blue
        (255, 225,  25),   # Yellow
        (245, 130,  48),   # Orange
        (145,  30, 180),   # Purple
        ( 70, 240, 240),   # Cyan
        (240,  50, 230),   # Magenta
        (210, 245,  60),   # Lime
        (250, 190, 212),   # Pink
        (  0, 128, 128),   # Teal
        (220, 190, 255),   # Lavender
        (170, 110,  40),   # Brown
        (255, 250, 200),   # Beige
        (128,   0,   0),   # Maroon
        (170, 255, 195),   # Mint
        (128, 128,   0),   # Olive
        (255, 215, 180),   # Apricot
        (  0,   0, 128),   # Navy
        (128, 128, 128),   # Grey
    ]

    def _get_distinct_color_rgb(self, index: int):
        """Return a distinct RGB tuple (0-255) for the given index, cycling if > palette size."""
        return self.DISTINCT_COLORS_RGB[index % len(self.DISTINCT_COLORS_RGB)]

    def _save_segmentation_masks(self, frame, masks_dict, tracked_objects, frame_idx, job_id):
        """Draw coloured mask overlays with label names directly on each food item using OpenCV.

        No individual mask images are saved – only a single annotated overlay image
        is produced and uploaded to S3.
        """
        from pathlib import Path

        # Create overlay directory (no separate masks directory needed any more)
        overlay_dir = self.config.OUTPUT_DIR / job_id / "masks_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # Work on a float copy so we can alpha-blend
        overlay = frame.astype(np.float32) / 255.0  # BGR float
        h, w = frame.shape[:2]

        # Assign a distinct colour to each object (BGR order for OpenCV)
        obj_ids_sorted = sorted(
            [oid for oid in masks_dict.keys() if oid in tracked_objects]
        )
        color_bgr_map = {}
        for idx, obj_id in enumerate(obj_ids_sorted):
            r, g, b = self._get_distinct_color_rgb(idx)
            color_bgr_map[obj_id] = (r, g, b)  # Frame is RGB, so store colors as RGB

        # --- draw coloured masks and collect label info ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        pad = 2  # padding around label text

        # (cx, cy, tx, ty, tw, th_text, baseline, display_label, bgr)
        # cx,cy = mask centroid (anchor);  tx,ty = label text origin (may be nudged)
        label_info = []

        for obj_id in obj_ids_sorted:
            mask = masks_dict[obj_id]
            label = tracked_objects[obj_id]['label']

            # Handle mask shape (could be 2D or 3D)
            if len(mask.shape) == 3:
                mask_2d = mask[0]
            else:
                mask_2d = mask

            mask_bool = mask_2d.astype(bool)
            if mask_bool.shape[:2] != (h, w):
                mask_bool = cv2.resize(
                    mask_2d.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Alpha-blend the colour onto the mask region (50 % opacity)
            bgr = color_bgr_map[obj_id]
            color_f = np.array([bgr[0] / 255.0, bgr[1] / 255.0, bgr[2] / 255.0])
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_bool,
                    overlay[:, :, c] * 0.5 + color_f[c] * 0.5,
                    overlay[:, :, c],
                )

            # Find mask centroid
            ys, xs = np.where(mask_bool)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
            else:
                cx, cy = w // 2, h // 2

            display_label = label[:40]
            (tw, th_text), baseline = cv2.getTextSize(display_label, font, font_scale, thickness)

            # Initial position centred on mask centroid, clamped to image bounds
            tx = max(0, min(cx - tw // 2, w - tw - pad * 2))
            ty = max(th_text + pad, min(cy, h - baseline - pad))

            label_info.append([cx, cy, tx, ty, tw, th_text, baseline, display_label, bgr])

        # --- nudge overlapping labels so they don't pile on top of each other ---
        def pill_rect(info):
            """Return (x1, y1, x2, y2) of the label pill for collision checks."""
            _, _, tx, ty, tw, th_text, baseline, _, _ = info
            return (tx - pad, ty - th_text - pad, tx + tw + pad, ty + baseline + pad)

        def rects_overlap(a, b):
            ax1, ay1, ax2, ay2 = pill_rect(a)
            bx1, by1, bx2, by2 = pill_rect(b)
            return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

        pill_h = label_info[0][5] + label_info[0][6] + pad * 2 + 2 if label_info else 14
        for i in range(len(label_info)):
            orig_ty = label_info[i][3]
            step = pill_h
            for attempt in range(30):
                if not any(rects_overlap(label_info[i], label_info[j]) for j in range(i)):
                    break
                # Alternate down / up from original position: +1, -1, +2, -2, …
                offset = ((attempt // 2) + 1) * step * (1 if attempt % 2 == 0 else -1)
                new_ty = orig_ty + offset
                if new_ty - label_info[i][5] - pad < 0 or new_ty + label_info[i][6] + pad > h:
                    continue
                label_info[i][3] = new_ty

        # Final image — coloured fills + text labels
        result = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

        # Draw label pills on the result image.
        # result is RGB; cv2 uses BGR so we convert, draw, then convert back.
        if label_info:
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            for info in label_info:
                cx, cy, tx, ty, tw, th_text, baseline, display_label, rgb = info
                # rgb stored as (r, g, b); cv2 needs (b, g, r)
                bgr_color = (rgb[2], rgb[1], rgb[0])
                pill_x1 = max(0, tx - pad)
                pill_y1 = max(0, ty - th_text - pad)
                pill_x2 = min(w - 1, tx + tw + pad)
                pill_y2 = min(h - 1, ty + baseline + pad)
                # White pill background for readability
                cv2.rectangle(result_bgr, (pill_x1, pill_y1), (pill_x2, pill_y2), (255, 255, 255), -1)
                # Label text in black for contrast on white background
                cv2.putText(result_bgr, display_label, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        overlay_filename = overlay_dir / f"frame_{frame_idx:05d}_all_masks.png"
        # Frame is RGB; use PIL to save so channels are stored correctly for web/mobile viewers
        Image.fromarray(result).save(str(overlay_filename))

        # Upload only the overlay to S3 (no individual masks)
        self._upload_segmented_images_to_s3(job_id, overlay_dir, frame_idx)

        logger.info(f"[{job_id}] Frame {frame_idx}: Saved labelled overlay to {overlay_filename}")
    
    def _upload_segmented_images_to_s3(self, job_id: str, overlay_dir: Path, frame_idx: int):
        """
        Upload the labelled overlay image to S3 for later retrieval.

        Structure: segmented_images/{job_id}/frame_XXXXX/overlays/all_masks.png
        """
        global s3_client

        if not S3_RESULTS_BUCKET or not UPLOAD_SEGMENTED_IMAGES:
            if not S3_RESULTS_BUCKET:
                logger.warning(
                    f"[{job_id}] S3_RESULTS_BUCKET not set, skipping S3 upload of segmented images. "
                    f"Results are saved locally at {overlay_dir}."
                )
            else:
                logger.info(f"[{job_id}] UPLOAD_SEGMENTED_IMAGES is disabled, skipping S3 upload of segmented images.")
            return

        try:
            # Initialize S3 client if not already done
            if s3_client is None:
                s3_client = boto3.client('s3')

            frame_folder = f"frame_{frame_idx:05d}"

            # Upload overlay file only (no individual masks)
            overlay_file = overlay_dir / f"frame_{frame_idx:05d}_all_masks.png"
            if overlay_file.exists():
                s3_key = f"segmented_images/{job_id}/{frame_folder}/overlays/all_masks.png"
                s3_client.upload_file(
                    str(overlay_file),
                    S3_RESULTS_BUCKET,
                    s3_key,
                    ExtraArgs={'ContentType': 'image/png'}
                )
                logger.info(f"[{job_id}] Uploaded overlay to s3://{S3_RESULTS_BUCKET}/{s3_key}")
            else:
                logger.warning(f"[{job_id}] Overlay file not found: {overlay_file}")

            logger.info(f"[{job_id}] Frame {frame_idx}: Uploaded labelled overlay to S3 (bucket: {S3_RESULTS_BUCKET}, path: segmented_images/{job_id}/{frame_folder}/)")

        except Exception as e:
            logger.error(f"[{job_id}] Failed to upload segmented images to S3: {e}", exc_info=True)
            # Don't fail the entire pipeline if S3 upload fails
    
    def _generate_segmented_video(self, video_path: Path, job_id: str, tracking_results: Dict):
        """
        After pipeline has results from the 5 frames, run the full 5-second video through SAM2
        with the detected labels/boxes to produce a segmented overlay video. Saves in the same
        directory as segmented images (masks_overlay) and uploads to S3 under segmented_images/{job_id}/.
        """
        objects = tracking_results.get('objects') or {}
        # Collect (obj_id, label, box) for objects that have box (from frame 0). Keys may be int or "ID{n}_{label}".
        initial_detections = []
        for key, data in objects.items():
            if not isinstance(data, dict):
                continue
            obj_id = data.get('obj_id')
            if obj_id is None:
                try:
                    if isinstance(key, int):
                        obj_id = key
                    else:
                        obj_id = int(str(key).replace("ID", "").split("_")[0])
                except (ValueError, TypeError, AttributeError):
                    continue
            box = data.get('box')
            label = data.get('label', '')
            if box is not None and len(box) == 4 and label:
                initial_detections.append((obj_id, label, box))
        if not initial_detections:
            logger.info(f"[{job_id}] No objects with boxes for segmented video; skipping")
            return
        # Read frames with cv2 (same coordinate space as initial_detections boxes).
        # cv2 ignores rotation metadata — the output video will have the rotation
        # tag copied from the original so iOS/Android players auto-rotate on playback.
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"[{job_id}] Could not open video for segmented overlay: {video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_duration_sec = getattr(self.config, "VIDEO_MAX_DURATION_SECONDS", 5.0)
        if total_frames / max(fps, 1) > max_duration_sec + 0.5:
            cap.release()
            logger.warning(f"[{job_id}] Video longer than {max_duration_sec}s; skipping segmented video")
            return

        # Detect rotation from the original video.
        # Write to stderr (unbuffered) so output appears in CloudWatch regardless of log level.
        # Primary method: parse tkhd transformation matrix from raw MP4 bytes.
        # We iterate ALL trak boxes because the audio tkhd (identity matrix) often comes first
        # and would incorrectly yield rotation=0 if we stop at the first tkhd found.
        video_rotation = 0
        try:
            import struct as _struct

            def _iter_mp4_boxes(data):
                """Yield (box_type_bytes, box_content_bytes) for boxes at this level."""
                i = 0
                while i + 8 <= len(data):
                    sz = int.from_bytes(data[i:i+4], 'big')
                    bt = data[i+4:i+8]
                    if sz < 8 or i + sz > len(data):
                        break
                    yield bt, data[i+8:i+sz]
                    i += sz

            def _tkhd_rotation(tkhd_data):
                """Return rotation (0/90/180/270) encoded in tkhd transformation matrix, or 0."""
                if not tkhd_data:
                    return 0
                ver = tkhd_data[0]
                # version(1)+flags(3) + timestamps+track_id+reserved+duration + reserved(8)+layer(2)+altgrp(2)+vol(2)+res(2)
                mbase = 40 if ver == 0 else 52  # v0: 4+20+16=40, v1: 4+32+16=52
                if mbase + 36 > len(tkhd_data):
                    return 0
                ma = _struct.unpack_from('>i', tkhd_data, mbase)[0]
                mb = _struct.unpack_from('>i', tkhd_data, mbase + 4)[0]
                mc = _struct.unpack_from('>i', tkhd_data, mbase + 12)[0]
                md = _struct.unpack_from('>i', tkhd_data, mbase + 16)[0]
                sys.stderr.write(f"[{job_id}] tkhd matrix: a={ma} b={mb} c={mc} d={md}\n")
                sys.stderr.flush()
                if ma == 0 and mb > 0 and mc < 0 and md == 0:
                    return 90
                elif ma == 0 and mb < 0 and mc > 0 and md == 0:
                    return 270
                elif ma < 0 and md < 0:
                    return 180
                return 0  # identity matrix (no rotation)

            _raw_bytes = video_path.read_bytes()
            # Walk moov → trak(s) → tkhd, check every tkhd for non-zero rotation
            for _bt0, _c0 in _iter_mp4_boxes(_raw_bytes):
                if _bt0 == b'moov':
                    for _bt1, _c1 in _iter_mp4_boxes(_c0):
                        if _bt1 == b'trak':
                            for _bt2, _c2 in _iter_mp4_boxes(_c1):
                                if _bt2 == b'tkhd':
                                    _rot = _tkhd_rotation(_c2)
                                    sys.stderr.write(f"[{job_id}] tkhd found, rotation={_rot}\n")
                                    sys.stderr.flush()
                                    if _rot != 0:
                                        video_rotation = _rot
                                        break
                            if video_rotation:
                                break
                    break  # only one moov box

        except Exception as _tkhd_err:
            sys.stderr.write(f"[{job_id}] tkhd rotation detection error: {_tkhd_err}\n")
            sys.stderr.flush()

        # Fallback: try ffmpeg -i stderr (catches stream-tag and display-matrix rotation sources)
        if not video_rotation:
            try:
                _ffinfo = subprocess.run(
                    ['ffmpeg', '-i', str(video_path)],
                    capture_output=True
                )
                _ffstderr = _ffinfo.stderr.decode('utf-8', errors='replace')
                for _line in _ffstderr.splitlines():
                    _ll = _line.lower().strip()
                    if _ll.startswith('rotate') and ':' in _ll:
                        try:
                            video_rotation = int(_ll.split(':')[1].strip())
                        except ValueError:
                            pass
                    elif 'displaymatrix' in _ll and 'rotation of' in _ll:
                        try:
                            _deg = float(_ll.split('rotation of')[1].split('degrees')[0].strip())
                            video_rotation = int(round(-_deg))
                        except (ValueError, IndexError):
                            pass
                sys.stderr.write(f"[{job_id}] ffmpeg stderr rotation: {video_rotation}\n")
                sys.stderr.flush()
            except Exception as _ff_err:
                sys.stderr.write(f"[{job_id}] ffmpeg rotation fallback error: {_ff_err}\n")
                sys.stderr.flush()

        sys.stderr.write(f"[{job_id}] Final video_rotation={video_rotation}\n")
        sys.stderr.flush()

        sample_fps = min(fps, 8.0)
        sample_step = max(1, round(fps / sample_fps))
        max_sampled = 40
        frames_list = []
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % sample_step == 0:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_h = int(self.config.RESIZE_WIDTH * aspect_ratio)
                frame_resized = cv2.resize(frame, (self.config.RESIZE_WIDTH, new_h))
                frames_list.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                if len(frames_list) >= max_sampled:
                    break
            frame_num += 1
        cap.release()
        logger.info(f"[{job_id}] Sampled {len(frames_list)} frames (fps={fps:.1f}, step={sample_step})")
        if not frames_list:
            logger.warning(f"[{job_id}] No frames read for segmented video")
            return
        # Bright saturated colors per object (BGR) — vivid enough to survive H.264 compression
        _PALETTE_BGR = [
            (0, 200, 255),   # yellow
            (0, 255, 100),   # green
            (255, 80,  80),  # blue
            (80,  80, 255),  # red
            (255, 0,  200),  # magenta
            (0,  220, 180),  # lime
            (200, 0,  255),  # purple
            (0,  180, 255),  # orange
        ]
        colors_bgr = {}
        for i, (obj_id, _, _) in enumerate(initial_detections):
            colors_bgr[obj_id] = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        obj_id_to_label = {det[0]: det[1] for det in initial_detections}

        h, w = frames_list[0].shape[:2]
        logger.info(f"[{job_id}] cv2 frame dimensions: {w}x{h} (rotation={video_rotation}° will be applied)")

        # Output video: same directory as segmented image overlays
        overlay_dir = self.config.OUTPUT_DIR / job_id / "masks_overlay"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        out_video_path = overlay_dir / "segmented_overlay_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_video_path), fourcc, sample_fps, (w, h))
        if not writer.isOpened():
            logger.warning(f"[{job_id}] Could not create video writer: {out_video_path}")
            return

        # Use SAM2 image predictor per frame — each frame is independent so there
        # is no temporal tracking drift. Food is nearly stationary so the initial
        # detection boxes work as prompts for every frame.
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        image_predictor = SAM2ImagePredictor(self.models.sam2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1

        for frame_idx, frame_rgb in enumerate(frames_list):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            overlay = frame_bgr.copy()

            try:
                with torch.inference_mode():
                    image_predictor.set_image(frame_rgb)

                    for obj_id, label, box in initial_detections:
                        x1, y1, x2, y2 = box
                        x1 = max(0.0, min(float(x1), w - 1))
                        y1 = max(0.0, min(float(y1), h - 1))
                        x2 = max(x1 + 1.0, min(float(x2), w))
                        y2 = max(y1 + 1.0, min(float(y2), h))

                        try:
                            masks, scores, _ = image_predictor.predict(
                                box=np.array([[x1, y1, x2, y2]]),
                                multimask_output=False,
                            )
                        except Exception as e:
                            logger.warning(f"[{job_id}] Frame {frame_idx}: SAM2 predict failed for {label}: {e}")
                            continue

                        if masks is None or len(masks) == 0:
                            continue

                        mask = masks[0].astype(bool)  # (H, W)
                        color = colors_bgr.get(obj_id, (128, 128, 128))

                        # Semi-transparent pixel-level mask overlay (75% color for visibility after H.264)
                        color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
                        color_layer[:] = color
                        blended = cv2.addWeighted(overlay, 0.25, color_layer, 0.75, 0)
                        overlay[mask] = blended[mask]
                        # Solid contour border so mask edge is always visible
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, color, 2)

                        # White pill label at mask centroid
                        if label:
                            ys, xs = np.where(mask)
                            if len(xs) > 0:
                                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                                (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                                pad = 4
                                px1 = max(0, cx - tw // 2 - pad)
                                py1 = max(0, cy - th // 2 - pad)
                                px2 = min(w, cx + tw // 2 + pad)
                                py2 = min(h, cy + th // 2 + pad)
                                cv2.rectangle(overlay, (px1, py1), (px2, py2), (255, 255, 255), -1)
                                cv2.putText(overlay, label, (px1 + pad, py2 - pad),
                                            font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            except Exception as e:
                logger.warning(f"[{job_id}] Frame {frame_idx}: SAM2 image prediction failed: {e}")

            writer.write(overlay)

        writer.release()
        logger.info(f"[{job_id}] Saved segmented overlay video (SAM2 per-frame, mp4v): {out_video_path}")

        # Re-encode from mp4v to H.264. Physically rotate frames using transpose so the
        # output plays correctly regardless of whether the player honours rotation metadata.
        #
        # iPhone portrait stores frames as LANDSCAPE with head at the RIGHT edge.
        # To display as portrait the frame must rotate 90° CCW → transpose=2.
        # rotate=90  (stream tag) or display_matrix rotation=-90 → transpose=2
        # rotate=270 (stream tag) or display_matrix rotation=-270/90 → transpose=1
        # rotate=180 → two 90° CCW passes
        _tmp = out_video_path.with_suffix('.raw.mp4')
        try:
            out_video_path.rename(_tmp)
            _vf_args = []
            if video_rotation == 90:
                _vf_args = ['-vf', 'transpose=2']   # 90° CCW — iPhone portrait (normal)
            elif video_rotation == 270 or video_rotation == -90:
                _vf_args = ['-vf', 'transpose=1']   # 90° CW  — iPhone portrait (upside-down)
            elif video_rotation == 180:
                _vf_args = ['-vf', 'transpose=2,transpose=2']  # 180°
            print(f"[{job_id}] Applying ffmpeg rotation filter: {_vf_args or 'none (0°)'} (rotation={video_rotation}°)")
            subprocess.run(
                ['ffmpeg', '-y',
                 '-i', str(_tmp),
                 *_vf_args,
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                 str(out_video_path)],
                check=True, capture_output=True
            )
            _tmp.unlink()
            logger.info(f"[{job_id}] Re-encoded overlay video to H.264 with physical rotation")
        except Exception as _enc_err:
            logger.warning(f"[{job_id}] FFmpeg H.264 re-encode failed ({_enc_err}); uploading mp4v")
            if _tmp.exists():
                _tmp.rename(out_video_path)
        # Upload to S3 (same prefix as segmented images)
        if S3_RESULTS_BUCKET and UPLOAD_SEGMENTED_IMAGES and out_video_path.exists():
            try:
                global s3_client
                if s3_client is None:
                    s3_client = boto3.client('s3')
                s3_key = f"segmented_images/{job_id}/segmented_overlay_video.mp4"
                s3_client.upload_file(
                    str(out_video_path),
                    S3_RESULTS_BUCKET,
                    s3_key,
                    ExtraArgs={'ContentType': 'video/mp4'}
                )
                logger.info(f"[{job_id}] Uploaded segmented video to s3://{S3_RESULTS_BUCKET}/{s3_key}")
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to upload segmented video to S3: {e}")
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Box format: [x1, y1, x2, y2]
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _run_florence2(self, task_prompt, text_input, image, processor, model):
        """Run Florence-2 inference"""
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # Use configurable generation parameters for longer, more detailed captions
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "max_new_tokens": self.config.FLORENCE2_MAX_NEW_TOKENS,
            "early_stopping": False,
            "use_cache": False,
        }
        
        # Add beam search for better quality (especially for captions)
        if self.config.FLORENCE2_NUM_BEAMS > 1:
            generation_kwargs["num_beams"] = self.config.FLORENCE2_NUM_BEAMS
        
        # Add sampling parameters if enabled (for more diverse outputs)
        if self.config.FLORENCE2_DO_SAMPLE:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.config.FLORENCE2_TEMPERATURE
        
        # Add min_length for caption and VQA tasks to encourage longer, more detailed outputs
        if task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
            generation_kwargs["min_length"] = self.config.FLORENCE2_MIN_LENGTH
        elif task_prompt == "<VQA>":
            # Use VQA-specific min_length for more complete answers
            generation_kwargs["min_length"] = self.config.FLORENCE2_VQA_MIN_LENGTH
        
        with torch.no_grad():
            generated_ids = model.generate(**generation_kwargs)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Debug: Log generated text for VQA tasks
        if task_prompt == "<VQA>":
            logger.debug(f"Generated text before post-processing: '{generated_text[:500]}'")
        
        parsed_answer = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        
        # Debug: Log parsed answer for VQA tasks
        if task_prompt == "<VQA>":
            logger.debug(f"Parsed answer after post-processing: {parsed_answer}")
        
        return parsed_answer
    
    def _match_objects(self, new_boxes, new_labels, tracked_objects):
        """Match new detections to existing tracked objects"""
        matched_mapping = {}
        unmatched_new = list(range(len(new_boxes)))
        
        if not tracked_objects:
            return matched_mapping, unmatched_new
        
        # Simple IoU matching
        for new_idx, new_box in enumerate(new_boxes):
            best_iou = 0
            best_id = None
            
            for obj_id, obj_data in tracked_objects.items():
                old_box = obj_data['box']
                iou = self._compute_iou(new_box, old_box)
                
                if iou > best_iou and iou >= self.config.IOU_MATCH_THRESHOLD:
                    best_iou = iou
                    best_id = obj_id
            
            if best_id is not None:
                matched_mapping[best_id] = new_idx
                unmatched_new.remove(new_idx)
        
        return matched_mapping, unmatched_new
    
    def _compute_iou(self, box1, box2):
        """Compute Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _estimate_depth_anything(self, frame_np):
        """
        Estimate relative depth using Depth Anything V2 Small.
        Returns a float32 numpy array (H x W) normalised to [0, 1]
        where 1 = closest to camera, 0 = furthest.
        """
        processor, model = self.models.depth_anything
        frame_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        inputs = processor(images=frame_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H, W)

        depth = predicted_depth.squeeze().cpu().numpy()

        # Resize to original frame resolution
        if depth.shape != frame_np.shape[:2]:
            depth = cv2.resize(depth, (frame_np.shape[1], frame_np.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]: higher value = closer to camera
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def _create_masked_depth_map(self, frame_np, depth_map, masks_dict, tracked_objects):
        """
        Overlay SAM2 masks onto the depth map to create a masked depth image.
        Background is dark; each food-item region shows its relative depth
        coloured with the viridis colourmap.

        Returns a PIL Image suitable for sending to Gemini.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h, w = depth_map.shape
        # RGBA canvas — transparent background
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        cmap = plt.get_cmap("viridis")
        for obj_id, mask in masks_dict.items():
            if mask is None:
                continue
            mask_bool = mask.astype(bool)
            # Map depth values through colourmap
            depth_vals = depth_map[mask_bool]  # float32 [0,1]
            colours = cmap(depth_vals)          # (N, 4) RGBA float
            rgba[mask_bool] = (colours * 255).astype(np.uint8)

        # Blend with a dim copy of the original frame for context
        original_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        # Darken non-masked area
        dim_bg = (original_rgb * 0.15).astype(np.uint8)
        output_rgb = dim_bg.copy()
        any_mask = rgba[:, :, 3] > 0
        output_rgb[any_mask] = rgba[any_mask, :3]

        return Image.fromarray(output_rgb)

    def _upload_depth_map_to_s3(self, job_id: str, frame_idx: int, depth_image: "Image"):
        """Upload masked depth map PNG to S3 under depth_maps/{job_id}/."""
        if not S3_RESULTS_BUCKET:
            logger.info(f"[{job_id}] S3_RESULTS_BUCKET not set — skipping depth map upload")
            return None
        try:
            import io
            buf = io.BytesIO()
            depth_image.save(buf, format="PNG")
            buf.seek(0)
            frame_folder = f"frame_{frame_idx:05d}"
            s3_key = f"depth_maps/{job_id}/{frame_folder}/masked_depth.png"
            s3.put_object(
                Bucket=S3_RESULTS_BUCKET,
                Key=s3_key,
                Body=buf,
                ContentType="image/png",
            )
            logger.info(f"[{job_id}] Uploaded depth map to s3://{S3_RESULTS_BUCKET}/{s3_key}")
            return s3_key
        except Exception as e:
            logger.error(f"[{job_id}] Failed to upload depth map: {e}")
            return None

    def _estimate_volume_from_depth_with_gemini(
        self,
        frame_np,
        depth_image,
        tracked_objects,
        plate_diameter_cm: float,
        job_id: str,
        user_context: dict = None,
    ) -> dict:
        """
        Second Gemini pass: send original image + masked depth map.
        Ask Gemini to estimate volume in ml per food ingredient.
        Returns {label: volume_ml} dict.
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.GEMINI_API_KEY)
        except Exception as e:
            logger.warning(f"[{job_id}] Gemini volume pass: init failed: {e}")
            return {}

        labels = [obj["label"] for obj in tracked_objects.values()]
        if not labels:
            return {}

        food_list = ", ".join(f'"{l}"' for l in labels)
        scale_note = (
            f"For scale reference, the plate/bowl in the image has an estimated diameter of "
            f"{plate_diameter_cm:.1f} cm."
            if plate_diameter_cm > 0
            else "No plate or bowl was detected; use your best visual judgement for scale."
        )

        prompt = (
            "You are given TWO images of the same food dish:\n"
            "  Image 1: the original photo taken by the user.\n"
            "  Image 2: a masked depth map of the same dish — brighter / yellow-green areas "
            "are closer to the camera, darker / purple areas are further away. Each coloured "
            "region corresponds to a separate food item that has been segmented.\n\n"
            f"{scale_note}\n\n"
            f"The detected food items are: {food_list}.\n\n"
            "Using the 3-D shape information visible in the depth map together with the "
            "original image, estimate the volume in millilitres (ml) for each food item. "
            "Consider the surface area of each coloured region and its apparent height/depth "
            "to compute a volume. Be realistic — a typical restaurant portion of rice is "
            "150–250 ml, a chicken breast 200–300 ml, a sauce 30–60 ml.\n\n"
        )
        prompt += self._build_user_context_suffix(user_context)
        prompt += (
            "\nReturn ONLY valid JSON — an array of objects:\n"
            '[{"name": "<food item>", "volume_ml": <number>}, ...]\n'
            "Include every item from the detected list. No extra text."
        )

        original_pil = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
        gemini_models = ["gemini-pro-latest", "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash"]
        response_text = ""
        for model_name in gemini_models:
            try:
                print(f"  Calling Gemini for volume estimation ({model_name})...")
                sys.stdout.flush()
                gm = genai.GenerativeModel(model_name, generation_config=self._GEMINI_GEN_CONFIG)
                response = gm.generate_content([prompt, original_pil, depth_image])
                response_text = response.text or ""
                if response_text:
                    break
            except Exception as e:
                logger.warning(f"[{job_id}] Gemini volume {model_name} failed: {e}")

        if not response_text:
            logger.warning(f"[{job_id}] Gemini volume estimation failed — using empty volumes")
            return {}

        # Parse JSON
        try:
            if "```json" in response_text:
                s = response_text.find("```json") + 7
                e_idx = response_text.find("```", s)
                json_str = response_text[s:e_idx].strip()
            elif "```" in response_text:
                s = response_text.find("```") + 3
                e_idx = response_text.find("```", s)
                json_str = response_text[s:e_idx].strip()
            else:
                s = response_text.find("[")
                e_idx = response_text.rfind("]") + 1
                json_str = response_text[s:e_idx] if s >= 0 else ""

            data = json.loads(json_str)
            volumes = {}
            for item in data:
                name = (item.get("name") or "").strip().lower()
                vol = item.get("volume_ml")
                if name and vol is not None:
                    volumes[name] = float(vol)
            logger.info(f"[{job_id}] Gemini volume estimates: {volumes}")
            return volumes
        except Exception as e:
            logger.warning(f"[{job_id}] Gemini volume JSON parse failed: {e}")
            return {}
    
    def _calibrate_from_reference_object(self, ref_box, depth_map_meters, frame_width, ref_type='plate'):
        """
        Calibrate pixel scale using reference object (plate or bowl) with known size
        
        Args:
            ref_box: Bounding box of reference object
            depth_map_meters: Depth map from Metric3D
            frame_width: Width of frame in pixels
            ref_type: 'plate' or 'bowl'
        
        Returns:
            (pixels_per_cm, reference_depth_m)
        """
        x1, y1, x2, y2 = [int(v) for v in ref_box]
        ref_width_px = x2 - x1
        ref_height_px = y2 - y1

        # Get reference diameter based on type
        if ref_type == 'plate':
            ref_diameter_cm = self.config.REFERENCE_PLATE_DIAMETER_CM
        elif ref_type == 'bowl':
            ref_diameter_cm = self.config.REFERENCE_BOWL_DIAMETER_CM
        else:
            ref_diameter_cm = self.config.REFERENCE_PLATE_DIAMETER_CM  # Default

        # Calculate potential calibration
        pixels_per_cm = ref_width_px / ref_diameter_cm

        # Validation: Check if this is a reasonable detection
        aspect_ratio = ref_width_px / max(ref_height_px, 1)
        is_reasonable = (
            0.6 < aspect_ratio < 1.5 and  # Roughly circular/oval
            ref_width_px > 50 and  # Not too small
            pixels_per_cm > 3.0 and  # Minimum reasonable scale
            pixels_per_cm < 30.0  # Maximum reasonable scale for 800px image
        )

        if not is_reasonable:
            logger.warning(f"{ref_type.capitalize()} detection unreliable (width={ref_width_px}px, aspect={aspect_ratio:.2f}, px/cm={pixels_per_cm:.2f})")
            # Use depth-based fallback calibration
            ref_region = depth_map_meters[y1:y2, x1:x2]
            avg_depth_m = np.median(ref_region[ref_region > 0])

            # Fallback: use general calibration value from config
            pixels_per_cm = self.config.DEFAULT_PIXELS_PER_CM
            logger.info(f"Using general calibration fallback: {pixels_per_cm:.2f} px/cm")

            return pixels_per_cm, avg_depth_m if avg_depth_m > 0 else 0.5

        # Reference object detection looks good, use it
        ref_region = depth_map_meters[y1:y2, x1:x2]
        avg_ref_depth_m = np.median(ref_region[ref_region > 0])

        logger.info(f"✓ {ref_type.capitalize()} calibration: {pixels_per_cm:.2f} px/cm, depth: {avg_ref_depth_m:.2f}m")
        return pixels_per_cm, avg_ref_depth_m
    
    def _calibrate_from_surface(self, surface_box, depth_map_meters, frame_width):
        """
        Calibrate reference plane depth from table/surface (no known size)
        
        Args:
            surface_box: Bounding box of surface/table
            depth_map_meters: Depth map from Metric3D
            frame_width: Width of frame in pixels
        
        Returns:
            reference_plane_depth_m
        """
        x1, y1, x2, y2 = [int(v) for v in surface_box]
        surface_region = depth_map_meters[y1:y2, x1:x2]
        valid_depths = surface_region[surface_region > 0]
        
        if len(valid_depths) > 0:
            # Use median depth of surface as reference plane
            reference_depth_m = np.median(valid_depths)
            logger.info(f"✓ Surface reference plane: {reference_depth_m:.3f}m")
        else:
            # Fallback: use median of entire scene, or config default
            scene_depths = depth_map_meters[depth_map_meters > 0]
            if len(scene_depths) > 0:
                reference_depth_m = np.median(scene_depths)
                logger.info(f"Using scene median as reference plane: {reference_depth_m:.3f}m")
            else:
                reference_depth_m = self.config.DEFAULT_REFERENCE_PLANE_DEPTH_M
                logger.warning(f"Using default reference plane depth: {reference_depth_m:.3f}m")
        
        return reference_depth_m
    
    def _calculate_volume_metric3d(self, mask, depth_map_meters, box, label):
        """Calculate volume using metric depth (implementation from test_tracking_metric3d.py)"""
        # This is a simplified version - full implementation in the original file
        mask_bool = mask.astype(bool)
        depth_values_m = depth_map_meters[mask_bool]
        
        if len(depth_values_m) == 0 or not self.calibration['calibrated']:
            return {'volume_ml': 0.0, 'avg_height_cm': 0.0, 'surface_area_cm2': 0.0}
        
        pixel_count = mask_bool.sum()
        pixels_per_cm = self.calibration['pixels_per_cm']
        surface_area_cm2 = pixel_count / (pixels_per_cm ** 2)
        
        valid_depths = depth_values_m[depth_values_m > 0]
        if len(valid_depths) == 0:
            return {'volume_ml': 0.0, 'avg_height_cm': 0.0, 'surface_area_cm2': surface_area_cm2}
        
        # Height calculation: Use reference plane (plate) as baseline
        # Calculate height relative to reference plane, not absolute depth difference
        reference_plane_depth_m = self.calibration.get('reference_plane_depth_m')
        
        if reference_plane_depth_m is not None and reference_plane_depth_m > 0:
            # Use reference plane approach: height = reference_plane_depth - object_top_depth
            # This gives us the height of the object above the plate/surface
            object_top_depth_m = np.percentile(valid_depths, 10)  # Top 10% = closest to camera (top of object)
            object_bottom_depth_m = np.percentile(valid_depths, 90)  # Bottom 90% = farthest (bottom/base)
            
            # Height is the difference from reference plane to top of object
            # If object is above reference plane, height is positive
            height_above_plane_m = reference_plane_depth_m - object_top_depth_m
            
            # Also check depth variation within object (for objects with significant height)
            depth_variation_m = object_bottom_depth_m - object_top_depth_m
            
            # Use the larger of: height above plane OR depth variation within object
            # This handles both cases: objects on plate vs objects with internal height
            raw_height_cm = max(height_above_plane_m, depth_variation_m) * 100
            raw_height_cm = max(0, raw_height_cm)  # Ensure non-negative
            
            logger.debug(f"Height calculation for {label}: reference_plane={reference_plane_depth_m:.3f}m, "
                        f"top={object_top_depth_m:.3f}m, height_above_plane={height_above_plane_m*100:.2f}cm, "
                        f"depth_variation={depth_variation_m*100:.2f}cm, final={raw_height_cm:.2f}cm")
        else:
            # Fallback: use depth variation within object (old method)
            base_depth_m = np.percentile(valid_depths, 75)  # Bottom of object
            top_depth_m = np.percentile(valid_depths, 15)   # Top of object
            depth_diff_m = base_depth_m - top_depth_m
            raw_height_cm = max(0, depth_diff_m * 100)
            logger.warning(f"No reference plane - using depth variation method for {label}")
        
        # General approach: Estimate reasonable height based on object size
        # Larger surface area → potentially taller object, but cap at reasonable limits
        # Use surface area to estimate if object is "flat" (fries) or "tall" (burger)
        
        # Estimate object diameter from surface area (assuming roughly circular)
        # diameter_cm ≈ 2 * sqrt(area / π)
        estimated_diameter_cm = 2 * np.sqrt(surface_area_cm2 / np.pi)
        
        # General height estimation: height should be proportional to size but capped
        # For food items, height is typically 10-30% of diameter for most items
        # Flat items (fries): 2-5% of diameter
        # Tall items (burgers): 20-40% of diameter
        
        label_lower = label.lower()
        
        # Detect flat items (fries, chips, etc.) - very low height-to-diameter ratio
        is_flat_item = any(word in label_lower for word in ['fries', 'chips', 'crisps', 'potato', 'flat'])
        
        if 'plate' in label_lower:
            height_cm = min(raw_height_cm, 2.5) if raw_height_cm > 5 else max(raw_height_cm, 1.5)
        elif any(word in label_lower for word in ['glass', 'cup']):
            height_cm = max(raw_height_cm, 8) if raw_height_cm < 3 else min(raw_height_cm, 15)
        elif is_flat_item:
            # Flat items: height should be very small relative to diameter
            height_cm = min(raw_height_cm, estimated_diameter_cm * 0.05)  # Max 5% of diameter
            height_cm = max(height_cm, 0.3)  # But at least 0.3cm
            height_cm = min(height_cm, 2.0)  # Cap at 2cm
        else:
            # General food items: height is typically 15-30% of diameter
            # But cap at reasonable maximums based on object size
            height_from_diameter = estimated_diameter_cm * 0.25  # 25% of diameter
            height_cm = min(raw_height_cm, height_from_diameter)
            # Cap based on absolute size: larger objects can be taller
            if estimated_diameter_cm < 5:  # Small items (<5cm diameter)
                height_cm = min(height_cm, 3.0)
            elif estimated_diameter_cm < 10:  # Medium items (5-10cm)
                height_cm = min(height_cm, 6.0)
            else:  # Large items (>10cm)
                height_cm = min(height_cm, 10.0)
            # Ensure minimum height
            height_cm = max(height_cm, 1.0)
        
        # General shape factor: accounts for irregular shapes and air gaps
        # Most food items are not perfect cylinders, so reduce volume
        # Smaller items tend to have more air gaps (lower factor)
        # Larger items are more solid (higher factor)
        
        if is_flat_item:
            shape_factor = 0.4  # Very irregular, lots of air
        elif estimated_diameter_cm < 5:
            shape_factor = 0.5  # Small items: more air gaps
        elif estimated_diameter_cm < 10:
            shape_factor = 0.6  # Medium items: moderate air gaps
        else:
            shape_factor = 0.65  # Large items: less air, more solid
        
        volume_ml = surface_area_cm2 * height_cm * shape_factor
        
        # Debug logging
        logger.info(f"Volume calculation for {label}: area={surface_area_cm2:.2f}cm², height={height_cm:.2f}cm, "
                   f"shape_factor={shape_factor:.2f}, diameter={estimated_diameter_cm:.2f}cm, "
                   f"raw_volume={volume_ml:.2f}ml")
        
        # Final validation: Cap volume at reasonable maximum
        # Calculate max reasonable volume based on diameter
        max_reasonable_volume_from_diameter = (estimated_diameter_cm ** 3) * 0.5
        
        # Also cap based on typical food volumes (stricter limits)
        typical_max_volumes = {
            'burger': 500, 'sandwich': 500, 'cheeseburger': 500, 'hamburger': 500,
            'fries': 200, 'french fries': 200, 'potato': 200,
            'pizza': 1000, 'salad': 500, 'soup': 500,
            'ice cream': 300, 'nugget': 100, 'chicken': 300
        }
        
        # Check if label matches any typical food
        matched_max = None
        for food_type, max_vol in typical_max_volumes.items():
            if food_type in label_lower:
                matched_max = max_vol
                break
        
        # Use the stricter limit (either diameter-based or food-specific)
        if matched_max:
            max_reasonable_volume = min(max_reasonable_volume_from_diameter, matched_max)
        else:
            max_reasonable_volume = max_reasonable_volume_from_diameter
        
        # Always cap at absolute maximum (1000ml) for safety
        max_reasonable_volume = min(max_reasonable_volume, 1000.0)
        
        if volume_ml > max_reasonable_volume:
            old_volume = volume_ml
            volume_ml = max_reasonable_volume
            logger.warning(f"⚠️ Volume capped from {old_volume:.1f}ml to {max_reasonable_volume:.1f}ml for '{label}' "
                          f"(diameter: {estimated_diameter_cm:.1f}cm, area: {surface_area_cm2:.1f}cm², height: {height_cm:.2f}cm)")
        
        # Note: Volume validation is now batched with estimation at the end (optimization)
        # We return the calculated volume as-is, validation happens later in batch
        
        return {
            'volume_ml': float(volume_ml),
            'avg_height_cm': float(height_cm),
            'surface_area_cm2': float(surface_area_cm2),
            'diameter_cm': float(estimated_diameter_cm)  # Store for later batch validation
        }
    
    def _validate_volume_with_gemini(self, food_name, calculated_volume_ml, height_cm, area_cm2, diameter_cm):
        """Use Gemini to validate if calculated volume is reasonable, return adjusted volume if needed"""
        try:
            import google.generativeai as genai
            import time
            # Add small delay to avoid rate limiting
            time.sleep(0.2)
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""You are a food portion estimation expert. Analyze this volume calculation:

Food: {food_name}
Calculated Volume: {calculated_volume_ml:.1f} ml
Surface Area: {area_cm2:.1f} cm²
Height: {height_cm:.2f} cm
Estimated Diameter: {diameter_cm:.1f} cm

Task: Check if this volume is reasonable for a TYPICAL RESTAURANT/HOME SERVING of this food.

❌ CORRECT these cases:
1. Measurement failures (height too low for vertical items, can't see inside containers)
2. Unrealistic portion sizes (way too large for typical serving)

Common serving sizes:
- Ribs: 200-400ml (2-4 ribs)
- Burger: 150-250ml (single burger)
- Fries: 150-300ml (side serving)
- Pasta: 300-500ml (main dish)
- Pizza slice: 200-300ml
- Chicken nuggets: 100-200ml (4-6 nuggets)
- Vegetables/sides: 100-200ml
- Drinks: 250-500ml
- Sauces/condiments: 30-100ml

✅ TRUST these volumes:
- Flat foods (pasta, pizza) with good measurements
- Multiple items combined (e.g., burger + toppings)
- Family-style portions (if explicitly multiple servings)

Respond ONLY with a JSON object:
{{
  "is_reasonable": true/false,
  "reason": "brief explanation",
  "suggested_volume_ml": number (only if unreasonable)
}}

Examples:
{{"is_reasonable": false, "reason": "1000ml too large for typical ribs serving (2-4 ribs = 200-400ml)", "suggested_volume_ml": 300}}
{{"is_reasonable": false, "reason": "Height 0.49cm too low for fries", "suggested_volume_ml": 150}}
{{"is_reasonable": true, "reason": "Reasonable pasta serving size"}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            import json
            # Handle markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            
            if not result.get('is_reasonable', True):
                suggested_volume = result.get('suggested_volume_ml')
                reason = result.get('reason', 'No reason provided')
                logger.info(f"Gemini volume validation for '{food_name}': {reason}")
                if suggested_volume and suggested_volume > 0:
                    return float(suggested_volume)
            
            # If reasonable or no suggestion, return original
            return calculated_volume_ml
            
        except Exception as e:
            logger.warning(f"Gemini volume validation failed: {e}")
            return calculated_volume_ml
    
    def _batch_validate_and_estimate_volumes_with_gemini(self, items_for_validation: list, untracked_items: list, job_id: str) -> dict:
        """
        Combined: Validate calculated volumes AND estimate untracked volumes in ONE Gemini call.
        Optimizes from N+M calls to 1 call.
        
        Args:
            items_for_validation: List of items with calculated volumes that need validation
            untracked_items: List of items without volumes that need estimation
            job_id: Job ID for logging
            
        Returns:
            Dict with 'validated' (obj_id -> validated_volume) and 'estimated' (obj_id -> estimated_volume)
        """
        if not self.config.GEMINI_API_KEY:
            # Fallback
            validated = {item['obj_id']: item['calculated_volume_ml'] for item in items_for_validation}
            estimated = {item['obj_id']: item['area_cm2'] * 2.0 for item in untracked_items}
            return {'validated': validated, 'estimated': estimated}
        
        if not items_for_validation and not untracked_items:
            return {'validated': {}, 'estimated': {}}
        
        try:
            import google.generativeai as genai
            import time
            import json
            time.sleep(0.2)  # Rate limiting
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Build prompt with both validation and estimation items
            validation_list = []
            for item in items_for_validation:
                validation_list.append(
                    f"- {item['label']}: calculated {item['calculated_volume_ml']:.1f}ml "
                    f"(height: {item['height_cm']:.2f}cm, area: {item['area_cm2']:.1f}cm², diameter: {item['diameter_cm']:.1f}cm)"
                )
            
            estimation_list = []
            for item in untracked_items:
                estimation_list.append(f"- {item['label']}: visible area {item['area_cm2']:.1f} cm²")
            
            prompt = f"""You are a food portion estimation expert. Perform TWO tasks:

1. **VALIDATE** calculated volumes - check if they're reasonable for typical servings
2. **ESTIMATE** volumes for items without calculations - provide typical serving volumes

Items to VALIDATE (calculated volumes):
{chr(10).join(validation_list) if validation_list else "None"}

Items to ESTIMATE (no volume calculated):
{chr(10).join(estimation_list) if estimation_list else "None"}

Common serving sizes:
- Ribs: 200-400ml (2-4 ribs)
- Burger: 150-250ml (single burger)
- Fries: 150-300ml (side serving)
- Pasta: 300-500ml (main dish)
- Pizza slice: 200-300ml
- Chicken nuggets: 100-200ml (4-6 nuggets)
- Vegetables/sides: 100-200ml
- Beans: 100-200ml (side serving)
- Gravy: 50-150ml (sauce serving)
- Mashed potatoes: 150-250ml (side serving)
- Drinks: 250-500ml
- Sauces/condiments: 30-100ml

For VALIDATION: If volume is unreasonable (too large/small, measurement failure), suggest corrected volume.
For ESTIMATION: Provide typical serving volume based on food type and visible area.

Respond ONLY with JSON:
{{
  "validated": [
    {{"food": "food_name", "validated_volume_ml": number, "reason": "explanation"}},
    ...
  ],
  "estimated": [
    {{"food": "food_name", "estimated_volume_ml": number, "reason": "explanation"}},
    ...
  ]
}}

If volume is reasonable, use the calculated volume. If no validation needed, omit from validated array.
Example:
{{
  "validated": [{{"food": "Ribs", "validated_volume_ml": 300, "reason": "Adjusted from 1000ml (too large)"}}],
  "estimated": [{{"food": "Beans", "estimated_volume_ml": 150, "reason": "Typical side serving"}}]
}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            
            # Map validated volumes
            validated_map = {}
            for item in items_for_validation:
                label = item['label']
                calculated_volume = item['calculated_volume_ml']
                
                # Find matching validation result
                matched = False
                for validated_item in result.get('validated', []):
                    if validated_item.get('food', '').lower() == label.lower():
                        validated_volume = float(validated_item.get('validated_volume_ml', calculated_volume))
                        validated_map[item['obj_id']] = validated_volume
                        if validated_volume != calculated_volume:
                            logger.info(f"[{job_id}] Gemini validated '{label}': {calculated_volume:.1f}ml → {validated_volume:.1f}ml - {validated_item.get('reason', '')}")
                        matched = True
                        break
                
                if not matched:
                    # No validation needed, use calculated volume
                    validated_map[item['obj_id']] = calculated_volume
            
            # Map estimated volumes
            estimated_map = {}
            for item in untracked_items:
                label = item['label']
                
                # Find matching estimation result
                matched = False
                for estimated_item in result.get('estimated', []):
                    if estimated_item.get('food', '').lower() == label.lower():
                        estimated_volume = float(estimated_item.get('estimated_volume_ml', 0))
                        estimated_map[item['obj_id']] = estimated_volume
                        logger.info(f"[{job_id}] Gemini estimated '{label}': {estimated_volume:.1f}ml - {estimated_item.get('reason', '')}")
                        matched = True
                        break
                
                if not matched:
                    # Fallback if no match
                    estimated_map[item['obj_id']] = item['area_cm2'] * 2.0
                    logger.warning(f"[{job_id}] No Gemini match for '{label}', using fallback")
            
            return {'validated': validated_map, 'estimated': estimated_map}
            
        except Exception as e:
            logger.warning(f"[{job_id}] Batch validate+estimate failed: {e}, using fallback")
            validated = {item['obj_id']: item['calculated_volume_ml'] for item in items_for_validation}
            estimated = {item['obj_id']: item['area_cm2'] * 2.0 for item in untracked_items}
            return {'validated': validated, 'estimated': estimated}
    
    def _batch_estimate_volumes_with_gemini(self, untracked_items: list, job_id: str) -> dict:
        """
        Batch estimate volumes for multiple untracked items in one Gemini call.
        Optimizes from N calls to 1 call.
        
        Args:
            untracked_items: List of dicts with 'obj_id', 'label', 'area_cm2'
            job_id: Job ID for logging
            
        Returns:
            Dict mapping obj_id -> estimated_volume_ml
        """
        if not self.config.GEMINI_API_KEY or not untracked_items:
            # Fallback: simple estimation
            return {item['obj_id']: item['area_cm2'] * 2.0 for item in untracked_items}
        
        try:
            import google.generativeai as genai
            import time
            import json
            time.sleep(0.2)  # Rate limiting
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Build prompt with all items
            items_list = []
            for item in untracked_items:
                items_list.append(f"- {item['label']}: visible area {item['area_cm2']:.1f} cm²")
            
            prompt = f"""You are a food portion estimation expert. Estimate typical serving volumes for these food items.

Food Items:
{chr(10).join(items_list)}

Task: Estimate reasonable TYPICAL RESTAURANT/HOME SERVING volumes in milliliters (ml) for each food.

Common portion ranges:
- Ribs: 200-400ml (2-4 ribs)
- Burger: 150-250ml (single burger)
- Fries: 150-300ml (side serving)
- Pasta: 300-500ml (main dish)
- Pizza slice: 200-300ml
- Chicken nuggets: 100-200ml (4-6 nuggets)
- Vegetables/sides: 100-200ml
- Beans: 100-200ml (side serving)
- Gravy: 50-150ml (sauce serving)
- Mashed potatoes: 150-250ml (side serving)
- Drinks: 250-500ml
- Sauces/condiments: 30-100ml

Respond ONLY with JSON array:
[
  {{"food": "food_name", "estimated_volume_ml": number, "reason": "brief explanation"}},
  ...
]

Example:
[{{"food": "Beans", "estimated_volume_ml": 150, "reason": "Typical side serving"}}, {{"food": "Gravy", "estimated_volume_ml": 100, "reason": "Standard gravy portion"}}]"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            results = json.loads(response_text)
            
            # Map results to obj_ids
            volume_map = {}
            for item in untracked_items:
                label = item['label']
                # Find matching result
                matched = False
                for result in results:
                    if result.get('food', '').lower() == label.lower():
                        volume_map[item['obj_id']] = float(result.get('estimated_volume_ml', 0))
                        logger.info(f"[{job_id}] Gemini volume estimate for '{label}': {result.get('estimated_volume_ml')}ml - {result.get('reason', '')}")
                        matched = True
                        break
                
                if not matched:
                    # Fallback if no match
                    volume_map[item['obj_id']] = item['area_cm2'] * 2.0
                    logger.warning(f"[{job_id}] No Gemini match for '{label}', using fallback")
            
            return volume_map
            
        except Exception as e:
            logger.warning(f"[{job_id}] Batch volume estimation failed: {e}, using fallback")
            return {item['obj_id']: item['area_cm2'] * 2.0 for item in untracked_items}
    
    def _estimate_volume_with_gemini(self, food_name: str, area_cm2: float, job_id: str) -> float:
        """
        Use Gemini to estimate typical serving volume for a food item when volume calculation failed.
        Then this volume will be used with RAG for nutrition analysis.
        
        Args:
            food_name: Name of the food item
            area_cm2: Surface area in cm² (from bounding box)
            job_id: Job ID for logging
            
        Returns:
            Estimated volume in ml
        """
        if not self.config.GEMINI_API_KEY:
            # Fallback: simple estimation from area
            estimated_height_cm = 2.0
            return area_cm2 * estimated_height_cm
        
        try:
            import google.generativeai as genai
            import time
            import json
            time.sleep(0.2)  # Rate limiting
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""You are a food portion estimation expert. Estimate the typical serving volume for this food item.

Food: {food_name}
Visible Surface Area: {area_cm2:.1f} cm²

Task: Estimate a reasonable TYPICAL RESTAURANT/HOME SERVING volume in milliliters (ml) for this food.

Consider:
- Typical serving sizes for this food type
- The visible area suggests approximate portion size
- Common portion ranges:
  * Ribs: 200-400ml (2-4 ribs)
  * Burger: 150-250ml (single burger)
  * Fries: 150-300ml (side serving)
  * Pasta: 300-500ml (main dish)
  * Pizza slice: 200-300ml
  * Chicken nuggets: 100-200ml (4-6 nuggets)
  * Vegetables/sides: 100-200ml
  * Beans: 100-200ml (side serving)
  * Gravy: 50-150ml (sauce serving)
  * Mashed potatoes: 150-250ml (side serving)
  * Drinks: 250-500ml
  * Sauces/condiments: 30-100ml

Respond ONLY with a JSON object:
{{
  "estimated_volume_ml": number,
  "reason": "brief explanation of the estimate"
}}

Example:
{{"estimated_volume_ml": 250, "reason": "Typical side serving of mashed potatoes"}}
{{"estimated_volume_ml": 150, "reason": "Standard serving of beans"}}
{{"estimated_volume_ml": 100, "reason": "Typical gravy portion"}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            estimated_volume = result.get('estimated_volume_ml', 0)
            reason = result.get('reason', 'No reason provided')
            
            if estimated_volume > 0:
                logger.info(f"[{job_id}] Gemini volume estimate for '{food_name}': {estimated_volume:.1f}ml - {reason}")
                return float(estimated_volume)
            else:
                # Fallback if Gemini returns invalid value
                logger.warning(f"[{job_id}] Gemini returned invalid volume for '{food_name}', using fallback")
                return area_cm2 * 2.0  # Fallback: area * 2cm height
            
        except Exception as e:
            logger.warning(f"[{job_id}] Gemini volume estimation failed for '{food_name}': {e}, using fallback")
            # Fallback: simple estimation from area
            return area_cm2 * 2.0  # area * 2cm height
    
    def _format_and_filter_with_gemini(self, boxes, labels, vqa_answer, job_id, frame_idx):
        """
        Combined: Format VQA answer and filter non-food items in one Gemini call.
        Optimizes from 2 calls to 1 call per detection frame.
        """
        try:
            import google.generativeai as genai
            import time
            import json
            time.sleep(0.2)  # Rate limiting
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Create list of detected items
            items_list = ", ".join([f'"{label}"' for label in labels])
            
            prompt = f"""You are analyzing detected objects from a food image. Perform two tasks:

1. Format the VQA answer: Extract only food item names from "{vqa_answer}", list them separated by commas.
2. Filter detected items: From the detected items with bounding boxes, identify which are ACTUAL FOOD or BEVERAGES.

VQA Answer: {vqa_answer}
Detected items with boxes: {items_list}

Rules:
- Include: Any food, ingredients, beverages, condiments
- Exclude: Text overlays (like "VQA", "question", "instruction"), UI elements, non-edible objects, utensils, plates, tables

Respond ONLY with JSON:
{{
  "formatted_foods": "comma-separated list of food names from VQA",
  "food_items_to_keep": ["item1", "item2", ...]
}}

Example:
{{"formatted_foods": "ribs, potatoes, beans, gravy, mashed potatoes", "food_items_to_keep": ["Ribs", "Potatoes", "Beans"]}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            food_items_to_keep = [item.lower() for item in result.get('food_items_to_keep', [])]
            formatted_foods = result.get('formatted_foods', '')
            
            # Filter boxes and labels based on Gemini's response
            filtered_boxes = []
            filtered_labels = []
            for box, label in zip(boxes, labels):
                if label.lower() in food_items_to_keep:
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
                else:
                    logger.info(f"[{job_id}] Frame {frame_idx}: Gemini filtered out non-food: '{label}'")
            
            # Return filtered boxes/labels and formatted foods (if needed)
            return filtered_boxes, filtered_labels, formatted_foods
            
        except Exception as e:
            logger.warning(f"[{job_id}] Frame {frame_idx}: Gemini format+filter failed: {e}, keeping all detections")
            return boxes.tolist() if hasattr(boxes, 'tolist') else boxes, labels, ""
    
    def _filter_non_food_with_gemini(self, boxes, labels, job_id, frame_idx):
        """Use Gemini to filter out non-food items from detected objects (legacy, kept for compatibility)"""
        filtered_boxes, filtered_labels, _ = self._format_and_filter_with_gemini(boxes, labels, "", job_id, frame_idx)
        return filtered_boxes, filtered_labels
    
    def _deduplicate_objects_with_gemini(self, tracking_results, job_id):
        """Use Gemini to merge duplicate objects detected across different frames with different labels"""
        if not self.config.GEMINI_API_KEY:
            return tracking_results
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Skip non-food items
            skip_keywords = [
                'question', 'vqa', 'text', 'plate', 'platter', 'fork', 'knife', 'spoon', 
                'glass', 'cup', 'mug', 'bottle', 'table', 'bowl', 'container'
            ]
            
            # Build list of all detected objects with their metadata
            objects_summary = []
            valid_obj_ids = []
            
            for obj_id, obj_data in tracking_results['objects'].items():
                label = obj_data['label']
                
                # Filter out non-food items
                if any(keyword in label.lower() for keyword in skip_keywords):
                    logger.info(f"[{job_id}] Filtering out non-food item: {label}")
                    continue
                
                volume = obj_data['statistics']['max_volume_ml']
                area = obj_data['statistics']['max_area_cm2']
                objects_summary.append(f"ID{obj_id}: {label} (volume: {volume:.1f}ml, area: {area:.1f}cm²)")
                valid_obj_ids.append((obj_id, obj_data))
            
            if len(valid_obj_ids) <= 1:
                # Nothing to deduplicate
                return tracking_results
            
            prompt = f"""Analyze this list of detected food objects from a video. Some objects may be the SAME physical item detected with different labels across frames. Identify which objects should be MERGED as duplicates.

Detected Objects:
{chr(10).join(objects_summary)}

Rules for merging:
- **MERGE** if labels refer to the SAME food type (e.g., "Ribs" and "Meat" are likely the same item, "Mashed Potatoes" and "Potatoes" are the same)
- **MERGE** if volumes/areas are very similar AND labels are related (e.g., two "Beans" with 50ml each = likely same)
- **KEEP SEPARATE** if labels are clearly different foods (e.g., "Ribs" vs "Beans", "Fries" vs "Cola")
- **KEEP SEPARATE** if same label but volumes are significantly different (different servings)

Respond ONLY with JSON listing merge groups:
{{
  "merge_groups": [
    {{"ids": ["ID1", "ID3"], "reason": "Both are meat/ribs"}},
    {{"ids": ["ID2", "ID5"], "reason": "Both are potatoes with similar volume"}}
  ],
  "keep_separate": ["ID4", "ID6"]
}}

If no duplicates, respond: {{"merge_groups": [], "keep_separate": ["ID1", "ID2", ...]}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            import json
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            decisions = json.loads(response_text)
            merge_groups = decisions.get('merge_groups', [])
            
            if not merge_groups:
                logger.info(f"[{job_id}] No duplicates to merge")
                return tracking_results
            
            # Apply merges
            merged_objects = {}
            merged_ids = set()
            
            for group in merge_groups:
                ids_to_merge = group['ids']
                reason = group.get('reason', 'duplicate')
                
                # Extract numeric IDs (e.g., "ID1" -> 1)
                numeric_ids = []
                for id_str in ids_to_merge:
                    try:
                        numeric_id = int(id_str.replace('ID', ''))
                        numeric_ids.append(numeric_id)
                    except ValueError:
                        continue
                
                if len(numeric_ids) < 2:
                    continue  # Need at least 2 to merge
                
                # Find the objects to merge
                objects_to_merge = []
                for obj_id, obj_data in valid_obj_ids:
                    if obj_id in numeric_ids:
                        objects_to_merge.append((obj_id, obj_data))
                
                if len(objects_to_merge) < 2:
                    continue
                
                # Merge: keep the one with highest volume
                objects_to_merge.sort(key=lambda x: x[1]['statistics']['max_volume_ml'], reverse=True)
                primary_id, primary_data = objects_to_merge[0]
                
                # Aggregate volumes and metadata
                total_volume = sum(obj[1]['statistics']['max_volume_ml'] for obj in objects_to_merge)
                max_area = max(obj[1]['statistics']['max_area_cm2'] for obj in objects_to_merge)
                max_height = max(obj[1]['statistics']['max_height_cm'] for obj in objects_to_merge)
                
                # Use the most descriptive label (longest or most specific)
                labels = [obj[1]['label'] for obj in objects_to_merge]
                best_label = max(labels, key=len)
                
                # Create merged object
                merged_data = primary_data.copy()
                merged_data['label'] = best_label
                merged_data['statistics']['max_volume_ml'] = total_volume
                merged_data['statistics']['mean_volume_ml'] = total_volume
                merged_data['statistics']['median_volume_ml'] = total_volume
                merged_data['statistics']['max_area_cm2'] = max_area
                merged_data['statistics']['max_height_cm'] = max_height
                
                merged_objects[primary_id] = merged_data
                
                for obj_id, _ in objects_to_merge:
                    merged_ids.add(obj_id)
                
                merged_labels = ', '.join(labels)
                logger.info(f"[{job_id}] ✓ Merged [{merged_labels}] → '{best_label}' (total: {total_volume:.1f}ml). Reason: {reason}")
            
            # Add non-merged objects
            for obj_id, obj_data in valid_obj_ids:
                if obj_id not in merged_ids:
                    merged_objects[obj_id] = obj_data
            
            tracking_results['objects'] = merged_objects
            return tracking_results
            
        except Exception as e:
            logger.warning(f"[{job_id}] Gemini deduplication failed: {e}", exc_info=True)
            return tracking_results
    
    def _deduplicate_and_combine_with_gemini(self, tracking_results, job_id):
        """
        Combined: Deduplicate objects and combine similar items in one Gemini call.
        Optimizes from 2 calls to 1 call.
        """
        if not self.config.GEMINI_API_KEY:
            return tracking_results
        
        try:
            import google.generativeai as genai
            import time
            import json
            time.sleep(0.2)  # Rate limiting
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Build list of all detected objects
            skip_keywords = [
                'question', 'vqa', 'text', 'plate', 'platter', 'fork', 'knife', 'spoon', 
                'glass', 'cup', 'mug', 'bottle', 'table', 'bowl', 'container'
            ]
            
            objects_summary = []
            valid_obj_ids = []
            
            for obj_id, obj_data in tracking_results['objects'].items():
                label = obj_data['label']
                
                # Filter out non-food items
                if any(keyword in label.lower() for keyword in skip_keywords):
                    continue
                
                volume = obj_data['statistics']['max_volume_ml']
                area = obj_data['statistics']['max_area_cm2']
                objects_summary.append(f"ID{obj_id}: {label} (volume: {volume:.1f}ml, area: {area:.1f}cm²)")
                valid_obj_ids.append((obj_id, obj_data))
            
            if len(valid_obj_ids) <= 1:
                return tracking_results
            
            prompt = f"""Analyze this list of detected food objects from a video and perform TWO tasks:

1. **Deduplicate**: Identify objects that are the SAME physical item with different labels (e.g., "Ribs" + "Meat" = same)
2. **Combine**: Identify small items that should be combined (garnishes, condiments, sauces)

Detected Objects:
{chr(10).join(objects_summary)}

Rules:
- **MERGE (Deduplicate)**: Same food type with different labels (e.g., "Ribs" and "Meat", "Mashed Potatoes" and "Potatoes")
- **COMBINE**: Small garnishes (parsley, herbs), condiments, sauces that are sprinkled/spread
- **KEEP SEPARATE**: Main dishes, distinct portions, different foods

Respond ONLY with JSON:
{{
  "merge_groups": [
    {{"ids": ["ID1", "ID3"], "reason": "Both are meat/ribs"}}
  ],
  "combine": ["item_name1", "item_name2"],
  "keep_separate": ["ID4", "ID6"]
}}

If no duplicates/combinations, respond: {{"merge_groups": [], "combine": [], "keep_separate": ["ID1", "ID2", ...]}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            decisions = json.loads(response_text)
            
            # Step 1: Apply merges (deduplication)
            merge_groups = decisions.get('merge_groups', [])
            merged_objects = {}
            merged_ids = set()
            
            for group in merge_groups:
                ids_to_merge = group['ids']
                reason = group.get('reason', 'duplicate')
                
                numeric_ids = []
                for id_str in ids_to_merge:
                    try:
                        numeric_id = int(id_str.replace('ID', ''))
                        numeric_ids.append(numeric_id)
                    except ValueError:
                        continue
                
                if len(numeric_ids) < 2:
                    continue
                
                objects_to_merge = [(obj_id, obj_data) for obj_id, obj_data in valid_obj_ids if obj_id in numeric_ids]
                
                if len(objects_to_merge) < 2:
                    continue
                
                # Merge: keep the one with highest volume
                objects_to_merge.sort(key=lambda x: x[1]['statistics']['max_volume_ml'], reverse=True)
                primary_id, primary_data = objects_to_merge[0]
                
                total_volume = sum(obj[1]['statistics']['max_volume_ml'] for obj in objects_to_merge)
                max_area = max(obj[1]['statistics']['max_area_cm2'] for obj in objects_to_merge)
                max_height = max(obj[1]['statistics']['max_height_cm'] for obj in objects_to_merge)
                labels = [obj[1]['label'] for obj in objects_to_merge]
                best_label = max(labels, key=len)
                
                merged_data = primary_data.copy()
                merged_data['label'] = best_label
                merged_data['statistics']['max_volume_ml'] = total_volume
                merged_data['statistics']['mean_volume_ml'] = total_volume
                merged_data['statistics']['median_volume_ml'] = total_volume
                merged_data['statistics']['max_area_cm2'] = max_area
                merged_data['statistics']['max_height_cm'] = max_height
                
                merged_objects[primary_id] = merged_data
                for obj_id, _ in objects_to_merge:
                    merged_ids.add(obj_id)
                
                logger.info(f"[{job_id}] ✓ Merged [{', '.join(labels)}] → '{best_label}' ({total_volume:.1f}ml). Reason: {reason}")
            
            # Add non-merged objects
            for obj_id, obj_data in valid_obj_ids:
                if obj_id not in merged_ids:
                    merged_objects[obj_id] = obj_data
            
            # Step 2: Apply combinations
            combine_items = [item.lower() for item in decisions.get('combine', [])]
            item_groups = {}
            for obj_id, obj_data in merged_objects.items():
                label = obj_data['label']
                if label not in item_groups:
                    item_groups[label] = []
                item_groups[label].append({
                    'obj_id': obj_id,
                    'volume': obj_data['statistics']['max_volume_ml'],
                    'data': obj_data
                })
            
            final_objects = {}
            combined_ids = set()
            
            for label, instances in item_groups.items():
                if label.lower() in combine_items and len(instances) > 1:
                    total_volume = sum(i['volume'] for i in instances)
                    first_instance = instances[0]
                    
                    combined_id = first_instance['obj_id']
                    final_objects[combined_id] = first_instance['data'].copy()
                    final_objects[combined_id]['statistics']['max_volume_ml'] = total_volume
                    final_objects[combined_id]['statistics']['mean_volume_ml'] = total_volume
                    final_objects[combined_id]['statistics']['median_volume_ml'] = total_volume
                    
                    for i in instances:
                        combined_ids.add(i['obj_id'])
                    
                    logger.info(f"[{job_id}] Combined {len(instances)} instances of '{label}' into 1 ({total_volume:.1f}ml total)")
                else:
                    for instance in instances:
                        final_objects[instance['obj_id']] = instance['data']
            
            tracking_results['objects'] = final_objects
            return tracking_results
            
        except Exception as e:
            logger.warning(f"[{job_id}] Gemini deduplicate+combine failed: {e}", exc_info=True)
            return tracking_results
    
    def _combine_similar_items(self, tracking_results):
        """Use Gemini to intelligently combine garnishes/small ingredients while keeping main dishes separate (legacy, kept for compatibility)"""
        if not self.config.GEMINI_API_KEY:
            return tracking_results  # Skip if no Gemini key
        
        try:
            import google.generativeai as genai
            import time
            time.sleep(0.2)  # Rate limiting
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Build list of items with their counts
            item_groups = {}
            for obj_id, obj_data in tracking_results['objects'].items():
                label = obj_data['label']
                volume = obj_data['statistics']['max_volume_ml']
                
                if label not in item_groups:
                    item_groups[label] = []
                item_groups[label].append({
                    'obj_id': obj_id,
                    'volume': volume,
                    'data': obj_data
                })
            
            # Ask Gemini which items should be combined
            items_summary = []
            for label, instances in item_groups.items():
                volumes_str = [f"{i['volume']:.1f}ml" for i in instances]
                items_summary.append(f"{label}: {len(instances)} instances, volumes: {volumes_str}")
            
            prompt = f"""Analyze this list of detected food items and decide which should be combined vs kept separate:

{chr(10).join(items_summary)}

Rules:
- **Combine**: Small garnishes (parsley, herbs), condiments, sauces - these are sprinkled/spread across the dish
- **Keep Separate**: Main dishes (burgers, pizzas, servings), distinct portions (multiple fries containers, multiple drinks)

Respond ONLY with JSON:
{{
  "combine": ["item_name1", "item_name2"],  // Items to combine (garnishes, small ingredients)
  "keep_separate": ["item_name3", "item_name4"]  // Items to keep as individual servings
}}

Example:
{{"combine": ["Parsley", "Basil", "Sauce"], "keep_separate": ["Hamburger", "Fries", "Cola"]}}"""

            response = gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            import json
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            decisions = json.loads(response_text)
            combine_items = [item.lower() for item in decisions.get('combine', [])]
            
            # Combine items that Gemini suggested
            new_objects = {}
            combined_ids = set()
            
            for label, instances in item_groups.items():
                if label.lower() in combine_items and len(instances) > 1:
                    # Combine all instances into one
                    total_volume = sum(i['volume'] for i in instances)
                    first_instance = instances[0]
                    
                    # Create combined entry
                    combined_id = first_instance['obj_id']
                    new_objects[combined_id] = first_instance['data'].copy()
                    new_objects[combined_id]['statistics']['max_volume_ml'] = total_volume
                    new_objects[combined_id]['statistics']['mean_volume_ml'] = total_volume
                    new_objects[combined_id]['statistics']['median_volume_ml'] = total_volume
                    
                    # Mark other instances as combined
                    for i in instances:
                        combined_ids.add(i['obj_id'])
                    
                    logger.info(f"Combined {len(instances)} instances of '{label}' into 1 ({total_volume:.1f}ml total)")
                else:
                    # Keep separate
                    for instance in instances:
                        new_objects[instance['obj_id']] = instance['data']
            
            # Update tracking results
            tracking_results['objects'] = new_objects
            return tracking_results
            
        except Exception as e:
            logger.warning(f"Gemini item combining failed: {e}, keeping items as-is")
            return tracking_results
    
    def _analyze_nutrition(self, tracking_results, job_id):
        """
        Nutrition analysis pipeline:
          1. Gemini volume (from depth pass) x FAO density -> mass_g
          2. USDA RAG -> kcal/100g -> total_kcal
          3. Fallback to Gemini detection grams if no volume available
        """
        logger.info(f"[{job_id}] Running nutrition analysis...")

        # Deduplicate and combine items
        if self.config.GEMINI_API_KEY:
            tracking_results = self._deduplicate_and_combine_with_gemini(tracking_results, job_id)

        rag = self.models.rag

        import re as _re
        # Use whole-word matching to avoid false positives like
        # 'mat' → 'kalamata', 'table' → 'vegetable', 'cup' → 'cupcake'
        skip_keywords = [
            'plate', 'platter', 'fork', 'knife', 'spoon', 'glass', 'cup', 'mug', 'bottle',
            'dining table', 'bowl', 'water', 'sprinkle', 'surface', 'wooden', 'board',
            'cutting board', 'background', 'setting', 'scene', 'some other', 'other objects',
            'container', 'napkin', 'tissue', 'placemat',
        ]
        _skip_re = _re.compile(
            r'\b(' + '|'.join(_re.escape(k) for k in skip_keywords) + r')\b'
        )

        nutrition_items = []
        total_food_volume = 0
        total_mass = 0
        total_calories = 0

        for item_key, item_data in tracking_results['objects'].items():
            try:
                label = item_data['label']
                max_volume = item_data['statistics']['max_volume_ml']

                # Skip non-food items (whole-word match to avoid false positives)
                if _skip_re.search(label.lower()):
                    logger.info(f"[{job_id}] Skipping non-food item: '{label}'")
                    continue

                quantity = item_data['statistics'].get('quantity', 1)
                if quantity is None or quantity < 1:
                    quantity = 1

                # Priority 1: Gemini volume estimate (from depth pass) + FAO density + USDA kcal
                gemini_volume_ml = item_data.get('gemini_volume_ml')
                if gemini_volume_ml and gemini_volume_ml > 0:
                    density, fao_match, fao_src = rag._get_density_with_match(label)
                    kcal_per_100g, usda_match, usda_src = rag._get_calories_with_match(label)
                    mass_g = gemini_volume_ml * density
                    total_kcal = (mass_g / 100.0) * kcal_per_100g
                    nutrition = {
                        'food_name': label,
                        'quantity': int(quantity),
                        'volume_ml': gemini_volume_ml,
                        'density_g_per_ml': density,
                        'density_source': fao_src,
                        'density_matched': fao_match,
                        'density_similarity': 1.0,
                        'mass_g': round(mass_g, 1),
                        'calories_per_100g': round(kcal_per_100g, 1),
                        'total_calories': round(total_kcal, 1),
                        'calorie_source': usda_src,
                        'calorie_matched': usda_match,
                        'calorie_similarity': 1.0,
                        'matched_food': label,
                    }
                    logger.info(
                        f"[{job_id}] '{label}': volume={gemini_volume_ml:.1f}ml "
                        f"density={density:.3f}[{fao_match}] mass={mass_g:.1f}g "
                        f"kcal={total_kcal:.0f}[{usda_match}] [depth+FAO+USDA]"
                    )

                else:
                    # Priority 2: fallback to Gemini detection grams + USDA kcal
                    gemini_grams_g = item_data['statistics'].get('gemini_grams_g')
                    nutrition = rag.get_nutrition_for_food(
                        label, max_volume, mass_g=gemini_grams_g, quantity=quantity
                    )
                    logger.info(
                        f"[{job_id}] '{label}': fallback mass={nutrition.get('mass_g', 0):.1f}g "
                        f"kcal={nutrition.get('total_calories', 0):.0f} [gemini_grams+USDA]"
                    )

                item_mass = float(nutrition.get('mass_g') or 0.0)
                item_calories = float(nutrition.get('total_calories') or 0.0)
                nutrition['mass_g'] = item_mass
                nutrition['total_calories'] = item_calories

                nutrition_items.append(nutrition)
                total_food_volume += float(nutrition.get('volume_ml') or max_volume)
                total_mass += item_mass
                total_calories += item_calories
            except Exception as e:
                logger.warning(f"[{job_id}] Skipping item '{item_key}' due to error: {e}", exc_info=True)
                continue
        
        # Collect unquantified ingredients from florence_detections
        all_unquantified = []
        for detection in self.florence_detections:
            if 'unquantified_ingredients' in detection:
                all_unquantified.extend(detection['unquantified_ingredients'])
        
        # Remove duplicates while preserving order
        unique_unquantified = []
        seen = set()
        for item in all_unquantified:
            if item.lower() not in seen:
                unique_unquantified.append(item)
                seen.add(item.lower())
        
        result = {
            'items': nutrition_items,
            'summary': {
                'total_food_volume_ml': total_food_volume,
                'total_mass_g': total_mass,
                'total_calories_kcal': total_calories,
                'num_food_items': len(nutrition_items)
            }
        }
        
        # Add unquantified ingredients if any were detected
        if unique_unquantified:
            result['unquantified_ingredients'] = unique_unquantified
        
        return result

