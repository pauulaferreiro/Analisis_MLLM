import re
import os
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any

import torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from logger_config import log, monitor_latency, latency_log
from metrics_monitor import ResourceMonitor, aggregate_metric_dicts

os.environ["HF_TOKEN"] = "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"

#MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
#MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
MODEL_ID = "mistral-community/pixtral-12b"

CATEGORIES = [
    "Fiction", "News", "Show", "Sports",
    "Cartoons", "Music/Dance", "Arts/Culture",
    "Social", "Education/Science",
    "Leisure hobbies"
]


def parse_llm_json(text: str) -> dict:
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
            else:
                raise ValueError("No se encontró un bloque JSON válido en la salida.")
        return json.loads(json_str, strict=False)
    except Exception as e:
        log.error(f"Error parseando JSON del modelo: {e}")
        return {
            "predicted_category": "Error",
            "confidence": 0,
            "top_3_categories": [],
            "reasoning": "Parse failed"
        }


class VideoClassifier:
    @monitor_latency
    def __init__(self):
        log.info(f"CARGANDO MODELO {MODEL_ID}")

        #CUANTIZACION
        
        # self.bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            fix_mistral_regex=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            #quantization_config=self.bnb_config,    #CUANTIZAR
            dtype=torch.bfloat16,  # SIN CUANTIZAR
            device_map="auto",  # auto -> distribuir inteligentemente en la VRAM
            attn_implementation="sdpa",
        )

    def parse_eit_metadata(self, xml_path: str) -> dict:
        eit_data = {
            "start_time": "Unknown",
            "duration": "Unknown",
            "running_status": "Unknown",
            "event_name": "Unknown",
            "parental_country": "Unknown",
            "parental_rating": "Unknown",
            "extended_text": "Sin descripción"
        }

        if not xml_path or not os.path.exists(xml_path):
            return eit_data

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            event = root.find(".//event")

            if event is not None:
                eit_data["start_time"] = event.get("start_time", "Unknown")
                eit_data["duration"] = event.get("duration", "Unknown")
                eit_data["running_status"] = event.get("running_status", "Unknown")

                extended_texts = []
                ext_descriptors = []

                for child in event:
                    if child.tag == "short_event_descriptor":
                        event_name_node = child.find("event_name")
                        if event_name_node is not None and event_name_node.text:
                            eit_data["event_name"] = event_name_node.text

                    elif child.tag == "parental_rating_descriptor":
                        country = child.find("country")
                        if country is not None:
                            eit_data["parental_country"] = country.get("country_code", "Unknown")
                            raw_rating = country.get("rating", "Unknown")

                            # Aplicamos las reglas de conversión --> ESTI EN 300 468 (PAG 97)
                            if raw_rating != "Unknown":
                                try:
                                    rating_int = int(raw_rating, 16)
                                    if rating_int == 0x1D:
                                        eit_data["parental_rating"] = f"{raw_rating} - Todos los públicos"
                                    elif 0x01 <= rating_int <= 0x0F:
                                        age = rating_int + 3
                                        eit_data["parental_rating"] = f"{raw_rating} - {age} años"
                                    elif 0x10 <= rating_int <= 0xFF:
                                        eit_data["parental_rating"] = f"{raw_rating} - Definido por el broadcaster"
                                    else:
                                        eit_data["parental_rating"] = f"{raw_rating} - No definido"
                                except ValueError:
                                    eit_data["parental_rating"] = raw_rating

                    elif child.tag == "extended_event_descriptor":
                        ext_descriptors.append(child)

                ext_descriptors.sort(key=lambda x: int(x.get("descriptor_number", 0)))
                for desc in ext_descriptors:
                    text_node = desc.find("text")
                    if text_node is not None and text_node.text:
                        extended_texts.append(text_node.text)

                if extended_texts:
                    eit_data["extended_text"] = "".join(extended_texts)

        except Exception as e:
            log.error(f"Error parseando XML {xml_path}: {e}")

        return eit_data

    def build_context(self, eit_xml_path: Optional[str]) -> str:
        eit_data = self.parse_eit_metadata(eit_xml_path)

        context_str = f"""
Transport Stream EIT Metadata:
- Título del Evento: {eit_data['event_name']}
- Hora de Inicio: {eit_data['start_time']}
- Duración: {eit_data['duration']}
- Estado de Emisión: {eit_data['running_status']}

Audiencia:
- País de Calificación: {eit_data['parental_country']}
- Calificación de Edad (Hex): {eit_data['parental_rating']}

Descripción Extendida de la Emisión:
{eit_data['extended_text']}
"""
        return context_str.strip()

    def build_system_prompt(self, input_mode: str, allowed_categories: Optional[List[str]] = None) -> str:
        active_categories = allowed_categories if allowed_categories else CATEGORIES
        categories_str = ", ".join(active_categories)

        # 1. Definir un diccionario con todas las definiciones
        all_definitions = {
            "Fiction": "Fictional narrative content such as movies, TV series, scripted drama, comedy, romance, thriller, or historical drama.",
            "News": "Programs reporting or discussing real-world events, including news broadcasts, interviews, debates, or news documentaries.",
            "Show": "Entertainment programs such as quizzes, contests, variety shows, or talk shows where participants interact or compete.",
            "Sports": "Programs focused on sports events or competitions, including live matches, sports highlights, or sports analysis.",
            "Cartoons": "Content designed for children or teenagers, such as cartoons, educational shows, or youth entertainment programs.",
            "Music/Dance": "Programs focused on music performances, concerts, opera, ballet, or dance.",
            "Arts/Culture": "Programs about culture or arts such as theatre, literature, cinema, visual arts, fashion, or media culture.",
            "Social": "Programs discussing social, political, or economic topics, including documentaries, interviews, debates, or reports.",
            "Education/Science": "Informative or educational programs about science, nature, technology, medicine, or factual documentaries.",
            "Leisure hobbies": "Lifestyle or hobby programs such as travel, cooking, fitness, gardening, DIY, shopping, or motoring."
        }

        # 2. Filtrar solo las definiciones permitidas
        allowed_definitions = "\n".join([f"- {cat}: {all_definitions[cat]}" for cat in active_categories])

        if input_mode == "TEXT_ONLY_EIT":
            instruction_1 = "Classify using ONLY the provided DVB EIT metadata."
        elif input_mode == "IMAGE_ONLY":
            instruction_1 = "Classify using ONLY the visual content of the provided frame."
        elif input_mode == "EPG_RESOLVER":
            instruction_1 = (
                f"CRITICAL: You MUST choose your final 'predicted_category' ONLY from this list: "
                f"[{categories_str}]. Use the provided DVB EIT metadata to decide which of these "
                f"candidates is the most accurate."
            )
        else:
            instruction_1 = "Classify using BOTH the visual content and the DVB EIT metadata."

        return f"""You are an expert DVB broadcast content classifier.

    Your task is to classify the broadcast content into exactly one DVB category, following the category definitions specified in the ETSI EN 300 468 standard.
    Input mode: {input_mode}

    Category definitions:
    {allowed_definitions}

    Allowed categories to choose from:
    {categories_str}

    Instructions:
    1. {instruction_1}
    2. Return ONLY a strict valid JSON object.
    3. Do not wrap the JSON in markdown.
    4. Do not include explanations outside the JSON.
    5. The field "predicted_category" must be EXACTLY ONE from the Allowed categories list.
    6. CRITICAL: Do NOT invent new categories.
    7. CRITICAL: The field "top_3_categories" MUST contain EXACTLY THREE categories from the Allowed list, ranked by likelihood, with percentages summing to 100.
    8. The field "reasoning" must be very short and explain the choice.

    Required JSON schema:
    {{
    "reasoning": "Brief explanation without line breaks",
    "predicted_category": "One of the allowed categories",
    "confidence": 0,
    "top_3_categories": [
        {{"category": "Category name", "percentage": 0}},
        {{"category": "Category name", "percentage": 0}},
        {{"category": "Category name", "percentage": 0}}
    ]
    }}"""

    def _normalize_prediction(self, json_data: dict, allowed_categories: Optional[List[str]] = None) -> dict:
        # Usar las permitidas si se pasan, si no, usar las globales
        valid_categories = allowed_categories if allowed_categories else CATEGORIES
        
        category = json_data.get("predicted_category", "Error")
        
        # Validar contra valid_categories, NO contra CATEGORIES global
        if category not in valid_categories:
            category = "Undefined"

        top_3 = json_data.get("top_3_categories", [])
        clean_top_3 = []

        if isinstance(top_3, list):
            for item in top_3[:3]:
                if not isinstance(item, dict):
                    continue

                cat = item.get("category", "Undefined")
                pct = item.get("percentage", 0)

                if cat not in valid_categories:
                    cat = "Undefined"

                try:
                    pct = float(pct)
                except Exception:
                    pct = 0.0

                clean_top_3.append({
                    "category": cat,
                    "percentage": pct
                })

        used_categories = {x["category"] for x in clean_top_3 if x["category"] in valid_categories}

        if category in valid_categories and category not in used_categories:
            clean_top_3.insert(0, {"category": category, "percentage": 0.0})
            used_categories.add(category)

        for cat in valid_categories:
            if len(clean_top_3) >= 3:
                break
            if cat not in used_categories:
                clean_top_3.append({
                    "category": cat,
                    "percentage": 0.0
                })
                used_categories.add(cat)

        clean_top_3 = clean_top_3[:3]

        return {
            "prediction": category,
            "confidence": json_data.get("confidence", 0),
            "reasoning": json_data.get("reasoning", ""),
            "top_3_categories": clean_top_3
        }

    def _generate(self, messages, image: Optional[Image.Image] = None) -> dict:
        # Genera el prompt con la "estructura" correcta para cada modelo
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if image is not None:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to("cuda")
        else:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to("cuda")

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False, # determinista
            )

        generated_tokens = outputs[0][input_len:]   # Recorte tokens de entrada
        decoded_text = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True                # No "contamine" el resultado
        )[0]

        return {
            "raw_output": decoded_text,
            "tokens_in": int(input_len),
            "tokens_out": int(generated_tokens.shape[-1]),
        }

    def _generate_with_metrics(self, messages, image: Optional[Image.Image] = None) -> dict:
        with ResourceMonitor(sample_interval=0.05) as monitor:
            gen = self._generate(messages, image=image)

        metrics = monitor.finalize()
        gen.update(metrics)
        return gen

    def _build_prediction_record(
        self,
        measured: Dict[str, Any],
        pred: Dict[str, Any],
        mode: str,
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        record = {
            "mode": mode,
            "prediction": pred["prediction"],
            "confidence": pred["confidence"],
            "reasoning": pred["reasoning"],
            "top_3_categories": pred["top_3_categories"],
            "raw_output": measured["raw_output"],
            "tokens_in": measured["tokens_in"],
            "tokens_out": measured["tokens_out"],

            "latency_total_ms": measured.get("latency_total_ms"),
            "latency_total_s": measured.get("latency_total_s"),

            "gpu_energy_j": measured.get("gpu_energy_j"),
            "cpu_energy_j": measured.get("cpu_energy_j"),
            "total_energy_j": measured.get("total_energy_j"),

            "gpu_power_avg_w": measured.get("gpu_power_avg_w"),
            "cpu_power_avg_w": measured.get("cpu_power_avg_w"),
            "gpu_power_max_w": measured.get("gpu_power_max_w"),

            "gpu_util_avg_pct": measured.get("gpu_util_avg_pct"),
            "cpu_process_avg_pct": measured.get("cpu_process_avg_pct"),

            "gpu_mem_avg_mb": measured.get("gpu_mem_avg_mb"),
            "gpu_mem_max_mb": measured.get("gpu_mem_max_mb"),

            "ram_process_avg_mb": measured.get("ram_process_avg_mb"),
        }

        if extra_fields:
            record.update(extra_fields)

        return record

    def _build_single_aggregate_metrics(self, measured: Dict[str, Any], num_valid_frames: int = 0) -> Dict[str, Any]:
        tokens_out = measured.get("tokens_out", 0) or 0
        latency_s = measured.get("latency_total_s", 0) or 0.0
        total_energy_j = measured.get("total_energy_j")

        return {
            "latency_total_ms": measured.get("latency_total_ms"),
            "latency_total_s": latency_s,

            "gpu_energy_j": measured.get("gpu_energy_j"),
            "cpu_energy_j": measured.get("cpu_energy_j"),
            "total_energy_j": total_energy_j,

            "energy_per_token_j": round(total_energy_j / tokens_out, 6) if total_energy_j is not None and tokens_out > 0 else None,
            "latency_per_token_s": round(latency_s / tokens_out, 6) if tokens_out > 0 else None,
            "tokens_per_second": round(tokens_out / latency_s, 6) if latency_s > 0 else None,

            "gpu_power_avg_w": measured.get("gpu_power_avg_w"),
            "cpu_power_avg_w": measured.get("cpu_power_avg_w"),
            "gpu_power_max_w": measured.get("gpu_power_max_w"),

            "gpu_util_avg_pct": measured.get("gpu_util_avg_pct"),
            "cpu_process_avg_pct": measured.get("cpu_process_avg_pct"),

            "gpu_mem_avg_mb": measured.get("gpu_mem_avg_mb"),
            "gpu_mem_max_mb": measured.get("gpu_mem_max_mb"),

            "ram_process_avg_mb": measured.get("ram_process_avg_mb"),

            "tokens_in_total": measured.get("tokens_in", 0),
            "tokens_out_total": measured.get("tokens_out", 0),
            "num_valid_frames": num_valid_frames,
        }

    def _combine_stage_metrics(self, first_metrics: Dict[str, Any], second_metrics: Dict[str, Any], num_valid_frames: int) -> Dict[str, Any]:
        first_latency = first_metrics.get("latency_total_s", 0) or 0.0
        second_latency = second_metrics.get("latency_total_s", 0) or 0.0
        total_latency = first_latency + second_latency

        first_gpu_j = first_metrics.get("gpu_energy_j", 0) or 0.0
        second_gpu_j = second_metrics.get("gpu_energy_j", 0) or 0.0
        first_cpu_j = first_metrics.get("cpu_energy_j", 0) or 0.0
        second_cpu_j = second_metrics.get("cpu_energy_j", 0) or 0.0

        total_gpu_j = first_gpu_j + second_gpu_j
        total_cpu_j = first_cpu_j + second_cpu_j
        total_energy_j = total_gpu_j + total_cpu_j

        tokens_in_total = (first_metrics.get("tokens_in_total", 0) or 0) + (second_metrics.get("tokens_in_total", 0) or 0)
        tokens_out_total = (first_metrics.get("tokens_out_total", 0) or 0) + (second_metrics.get("tokens_out_total", 0) or 0)

        def weighted_avg(metric_key: str):
            weighted_sum = 0.0
            weight_total = 0.0

            for value, weight in [
                (first_metrics.get(metric_key), first_latency),
                (second_metrics.get(metric_key), second_latency),
            ]:
                if value is not None and weight > 0:
                    weighted_sum += value * weight
                    weight_total += weight

            return (weighted_sum / weight_total) if weight_total > 0 else None

        def max_non_null(metric_key: str):
            vals = [first_metrics.get(metric_key), second_metrics.get(metric_key)]
            vals = [v for v in vals if v is not None]
            return max(vals) if vals else None

        return {
            "latency_total_ms": int(total_latency * 1000),
            "latency_total_s": round(total_latency, 6),

            "gpu_energy_j": round(total_gpu_j, 6),
            "cpu_energy_j": round(total_cpu_j, 6),
            "total_energy_j": round(total_energy_j, 6),

            "energy_per_token_j": round(total_energy_j / tokens_out_total, 6) if tokens_out_total > 0 else None,
            "latency_per_token_s": round(total_latency / tokens_out_total, 6) if tokens_out_total > 0 else None,
            "tokens_per_second": round(tokens_out_total / total_latency, 6) if total_latency > 0 else None,

            "gpu_power_avg_w": round(total_gpu_j / total_latency, 6) if total_latency > 0 else None,
            "cpu_power_avg_w": round(total_cpu_j / total_latency, 6) if total_latency > 0 else None,
            "gpu_power_max_w": round(max_non_null("gpu_power_max_w"), 6) if max_non_null("gpu_power_max_w") is not None else None,

            "gpu_util_avg_pct": round(weighted_avg("gpu_util_avg_pct"), 4) if weighted_avg("gpu_util_avg_pct") is not None else None,
            "cpu_process_avg_pct": round(weighted_avg("cpu_process_avg_pct"), 4) if weighted_avg("cpu_process_avg_pct") is not None else None,

            "gpu_mem_avg_mb": round(weighted_avg("gpu_mem_avg_mb"), 4) if weighted_avg("gpu_mem_avg_mb") is not None else None,
            "gpu_mem_max_mb": round(max_non_null("gpu_mem_max_mb"), 4) if max_non_null("gpu_mem_max_mb") is not None else None,

            "ram_process_avg_mb": round(weighted_avg("ram_process_avg_mb"), 4) if weighted_avg("ram_process_avg_mb") is not None else None,

            "tokens_in_total": tokens_in_total,
            "tokens_out_total": tokens_out_total,
            "num_valid_frames": num_valid_frames,
        }

    @monitor_latency
    def classify_text_only(
        self,
        eit_xml_path: str,
        sample_name: str = "UNKNOWN",
        input_mode: str = "TEXT_ONLY_EIT",
        allowed_categories: Optional[List[str]] = None,
        extra_context: str = ""
    ):
        context_str = self.build_context(eit_xml_path)
        final_context = f"{extra_context}\n\nContext Data:\n{context_str}".strip()

        log.info("=" * 60)
        log.info(f"DEBUG: TEXTO PARA MODELO ({input_mode})")
        log.info("=" * 60)
        log.info(final_context)
        log.info("=" * 60)

        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    input_mode=input_mode,
                    allowed_categories=allowed_categories
                )
            },
            {
                "role": "user",
                "content": f"{final_context}\n\nClassify the broadcast category."
            }
        ]

        measured = self._generate_with_metrics(messages, image=None)
        json_data = parse_llm_json(measured["raw_output"])
        pred = self._normalize_prediction(json_data,allowed_categories=allowed_categories)

        log.info(
            f"{sample_name} | {input_mode} | Latencia: {measured['latency_total_s']:.3f}s | "
            f"Predicción: {pred['prediction']}"
        )
        latency_log.info(
            f"{sample_name},inference_text,{measured['latency_total_s']:.6f},{pred['prediction']}"
        )

        prediction_data = self._build_prediction_record(
            measured=measured,
            pred=pred,
            mode=input_mode.lower(),
        )

        aggregate_metrics = self._build_single_aggregate_metrics(measured, num_valid_frames=0)

        result = {
            "prediction": pred["prediction"],
            "top_3_categories": pred["top_3_categories"],
            "confidence": pred["confidence"],
            "reasoning": pred["reasoning"],
            "pipeline_mode": input_mode.lower()
        }

        torch.cuda.empty_cache()
        
        return result, [prediction_data], aggregate_metrics

    @monitor_latency
    def classify_frames(
        self,
        image_paths: List[str],
        eit_xml_path: str = None,
        sample_name: str = "UNKNOWN",
        frames_only: bool = False,
        allowed_categories: Optional[List[str]] = None
    ):
        if frames_only:
            input_mode = "IMAGE_ONLY"

            # Para la Fase 2 de EPG -> Visual
            if allowed_categories:
                user_text = (
                    "Analyze the image and classify the broadcast category based ONLY on visual features. "
                    f"You must choose only from these categories: {', '.join(allowed_categories)}."
                )
                log.info("=" * 60)
                log.info("DEBUG: MODO VISUAL RESTRINGIDO ACTIVADO (Sin metadatos EPG)")
                log.info(f"DEBUG: Categorías permitidas: {allowed_categories}")
                log.info("=" * 60)

            # Para la Fase 1 de Visual -> EPG  // Sólo visual
            else:
                user_text = "Analyze the image and classify the broadcast category based ONLY on visual features."
                log.info("=" * 60)
                log.info("DEBUG: MODO VISUAL ACTIVADO (Sin metadatos EPG)")
                log.info("=" * 60)

        # Para la opción Multimodal
        else:
            input_mode = "IMAGE_PLUS_EIT"
            context_str = self.build_context(eit_xml_path)
            user_text = (
                f"Context Data:\n{context_str}\n\n"
                f"Analyze the image and metadata, then classify the broadcast category."
            )

        log.info(f"Analizando modelo: {MODEL_ID}")
        log.info(f"Iniciando análisis de {len(image_paths)} frames.")

        votes: List[str] = []
        frame_predictions: List[Dict[str, Any]] = []
        metric_list: List[Dict[str, Any]] = []

        for i, img_path in enumerate(image_paths):
            try:
                target_image = Image.open(img_path).convert("RGB")
                messages = [
                    {
                        "role": "system",
                        "content": self.build_system_prompt(
                            input_mode=input_mode,
                            allowed_categories=allowed_categories
                        )
                    },
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
                ]

                measured = self._generate_with_metrics(messages, image=target_image)
                json_data = parse_llm_json(measured["raw_output"])
                pred = self._normalize_prediction(json_data,allowed_categories=allowed_categories)

                if pred["prediction"] != "Undefined":
                    votes.append(pred["prediction"])
                metric_list.append(measured)

                frame_name = Path(img_path).stem

                log.info(
                    f"{sample_name} | {frame_name} FRAME {i + 1}/{len(image_paths)} | "
                    f"Latencia: {measured['latency_total_s']:.3f}s | Predicción: {pred['prediction']}"
                )
                latency_log.info(
                    f"{sample_name},inference_frame,{i + 1},{measured['latency_total_s']:.6f},{pred['prediction']}"
                )

                frame_predictions.append(
                    self._build_prediction_record(
                        measured=measured,
                        pred=pred,
                        mode=input_mode.lower(),
                        extra_fields={
                            "frame_index": i + 1,
                            "frame_path": img_path,
                            "allowed_categories": allowed_categories if allowed_categories else CATEGORIES,
                        }
                    )
                )

            except Exception as e:
                log.error(f"Error procesando frame {img_path}: {e}")
                continue
            finally:
                torch.cuda.empty_cache()

        if not frame_predictions:
            log.warning("No se pudieron obtener predicciones de los frames.")
            empty_metrics = aggregate_metric_dicts([])
            return {
                "prediction": "Undefined",
                "top_3_categories": [],
                "confidence": None,
                "reasoning": "No valid frame predictions",
                "pipeline_mode": "image_only" if frames_only else "image_plus_eit"
            }, frame_predictions, empty_metrics

        if votes:
            winner, count = Counter(votes).most_common(1)[0]
        else:
            winner = "Undefined"
            count = 0

        aggregate_metrics = aggregate_metric_dicts(metric_list)

        log.info(
            f"{sample_name} | FASE VISUAL TERMINADA: Ganador temporal {winner} "
            f"({count}/{len(votes)} votos)"
        )
        latency_log.info(
            f"{sample_name},classify_frames_total,{aggregate_metrics['latency_total_s']:.6f},frames={len(image_paths)}"
        )

        return {
            "prediction": winner,
            "top_3_categories": [],
            "confidence": None,
            "reasoning": "Majority vote from frames",
            "pipeline_mode": "image_only" if frames_only else "image_plus_eit"
        }, frame_predictions, aggregate_metrics

    def classify_two_stage(
        self,
        eit_xml_path: str,
        image_paths: List[str],
        sample_name: str = "UNKNOWN",
    ):
        log.info(f"=== INICIANDO PIPELINE DE DOS FASES (VISUAL -> EPG) para {sample_name} ===")

        # FASE 1: Análisis visual
        log.info("--- FASE 1: Extracción Visual con Frames ---")
        visual_result, frame_predictions, visual_metrics = self.classify_frames(
            image_paths=image_paths,
            eit_xml_path=None,
            sample_name=f"{sample_name}_STAGE1",
            frames_only=True
        )

        visual_winner = visual_result.get("prediction", "Undefined")

        category_scores = defaultdict(float)
        for frame_data in frame_predictions:
            for cat_info in frame_data.get("top_3_categories", []):
                cat = cat_info.get("category")
                pct = cat_info.get("percentage", 0)
                if cat in CATEGORIES:
                    category_scores[cat] += pct

            main_pred = frame_data.get("prediction")
            if main_pred in CATEGORIES:
                category_scores[main_pred] += 20.0

        sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        top_3_candidates = [cat for cat, _score in sorted_cats[:3]]

        if not top_3_candidates:
            top_3_candidates = CATEGORIES[:3]

        log.info(f"Top 3 Visual extraído de los frames: {top_3_candidates}")

        # FASE 2: Resolución final con EPG
        log.info("--- FASE 2: Resolución de EPG ---")
        extra_context = (
            "Visual analysis across multiple frames suggests the content is most likely one of the "
            f"following: {', '.join(top_3_candidates)}."
        )

        final_result, text_predictions, text_metrics = self.classify_text_only(
            eit_xml_path=eit_xml_path,
            sample_name=f"{sample_name}_STAGE2",
            input_mode="EPG_RESOLVER",
            allowed_categories=top_3_candidates,
            extra_context=extra_context
        )

        final_winner = final_result["prediction"]
        final_metrics = self._combine_stage_metrics(
            visual_metrics,
            text_metrics,
            num_valid_frames=visual_metrics.get("num_valid_frames", len(image_paths))
        )

        log.info(
            f"=== RESULTADO FINAL {sample_name}: {final_winner} "
            f"(Resuelto desde EPG con base visual; ganador visual previo: {visual_winner}) ==="
        )

        return {
            "prediction": final_result["prediction"],
            "top_3_categories": final_result["top_3_categories"],
            "confidence": final_result["confidence"],
            "reasoning": final_result["reasoning"],
            "visual_winner": visual_winner,
            "pipeline_mode": "visual_then_epg"
        }, frame_predictions + text_predictions, final_metrics

    def classify_two_stage_epg_first(
        self,
        eit_xml_path: str,
        image_paths: List[str],
        sample_name: str = "UNKNOWN",
    ):
        log.info(f"=== INICIANDO PIPELINE DE DOS FASES (EPG -> VISUAL) para {sample_name} ===")

        # FASE 1: Análisis solo EPG para extraer top 3
        log.info("--- FASE 1: Extracción de Top 3 desde EPG ---")
        epg_result, text_predictions, epg_metrics = self.classify_text_only(
            eit_xml_path=eit_xml_path,
            sample_name=f"{sample_name}_STAGE1",
            input_mode="TEXT_ONLY_EIT"
        )

        top_3_candidates = [
            item["category"]
            for item in epg_result.get("top_3_categories", [])
            if item.get("category") in CATEGORIES
        ]

        if len(top_3_candidates) < 3:
            for cat in CATEGORIES:
                if cat not in top_3_candidates:
                    top_3_candidates.append(cat)
                if len(top_3_candidates) == 3:
                    break

        log.info(f"Top 3 EPG extraído: {top_3_candidates}")

        # FASE 2: Clasificación visual restringida a ese top 3
        log.info("--- FASE 2: Resolución visual restringida por EPG ---")
        visual_result, frame_predictions, visual_metrics = self.classify_frames(
            image_paths=image_paths,
            eit_xml_path=None,
            sample_name=f"{sample_name}_STAGE2",
            frames_only=True,
            allowed_categories=top_3_candidates
        )

        visual_winner = visual_result.get("prediction", "Undefined")

        visual_vote_counter = Counter()
        visual_score_counter = defaultdict(float)

        for frame_data in frame_predictions:
            pred_cat = frame_data.get("prediction")
            if pred_cat in top_3_candidates:
                visual_vote_counter[pred_cat] += 1
                visual_score_counter[pred_cat] += 100.0

            for cat_info in frame_data.get("top_3_categories", []):
                cat = cat_info.get("category")
                pct = cat_info.get("percentage", 0)
                if cat in top_3_candidates:
                    try:
                        visual_score_counter[cat] += float(pct)
                    except Exception:
                        pass

        if visual_score_counter:
            sorted_visual = sorted(visual_score_counter.items(), key=lambda x: x[1], reverse=True)
            final_winner = sorted_visual[0][0]
        elif visual_vote_counter:
            final_winner = visual_vote_counter.most_common(1)[0][0]
        else:
            final_winner = top_3_candidates[0]

        final_top_3 = []
        total_score = sum(visual_score_counter.values())

        if total_score > 0:
            sorted_visual = sorted(visual_score_counter.items(), key=lambda x: x[1], reverse=True)[:3]
            for cat, score in sorted_visual:
                pct = round((score / total_score) * 100, 2)
                final_top_3.append({"category": cat, "percentage": pct})
        else:
            for i, cat in enumerate(top_3_candidates[:3]):
                final_top_3.append({"category": cat, "percentage": 100.0 if i == 0 else 0.0})

        final_metrics = self._combine_stage_metrics(
            epg_metrics,
            visual_metrics,
            num_valid_frames=visual_metrics.get("num_valid_frames", len(image_paths))
        )

        log.info(
            f"=== RESULTADO FINAL {sample_name}: {final_winner} "
            f"(Resuelto visualmente dentro del top3 extraído de EPG) ==="
        )

        return {
            "prediction": final_winner,
            "top_3_categories": final_top_3,
            "confidence": epg_result.get("confidence", 0),
            "reasoning": (
                f"EPG reduced candidates to {top_3_candidates}; "
                f"final decision made by restricted visual voting."
            ),
            "epg_top_3_candidates": top_3_candidates,
            "epg_prediction": epg_result.get("prediction"),
            "visual_winner": visual_winner,
            "pipeline_mode": "epg_then_visual"
        }, text_predictions + frame_predictions, final_metrics

    def classify(
        self,
        eit_xml_path: str = None,
        image_paths: Optional[List[str]] = None,
        text_only: bool = False,
        frames_only: bool = False,
        two_stage: bool = True,
        two_stage_mode: str = "visual_then_epg",
        sample_name: str = "UNKNOWN",
    ):
        if text_only and frames_only:
            raise ValueError("No se puede activar text_only y frames_only a la vez.")

        if text_only or not image_paths:
            return self.classify_text_only(
                eit_xml_path=eit_xml_path,
                sample_name=sample_name,
            )

        if frames_only or not eit_xml_path:
            return self.classify_frames(
                image_paths=image_paths,
                eit_xml_path=eit_xml_path,
                sample_name=sample_name,
                frames_only=True,
            )

        if two_stage:
            if two_stage_mode == "visual_then_epg":
                return self.classify_two_stage(
                    eit_xml_path=eit_xml_path,
                    image_paths=image_paths,
                    sample_name=sample_name
                )
            elif two_stage_mode == "epg_then_visual":
                return self.classify_two_stage_epg_first(
                    eit_xml_path=eit_xml_path,
                    image_paths=image_paths,
                    sample_name=sample_name
                )
            else:
                raise ValueError(
                    f"Modo two_stage no válido: {two_stage_mode}. "
                    "Usa 'visual_then_epg' o 'epg_then_visual'."
                )

        return self.classify_frames(
            image_paths=image_paths,
            eit_xml_path=eit_xml_path,
            sample_name=sample_name,
            frames_only=False,
        )
