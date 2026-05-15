import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

from pipeline_docker import VideoClassifier
from logger_config import log

VALID_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "si", "sí"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Valor booleano no válido")


def find_frame_folders(frames_root_dir: Path, frames_subdir: str):
    samples = []

    for sample_dir in sorted(frames_root_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        frames_folder = sample_dir / frames_subdir

        if frames_folder.exists() and frames_folder.is_dir():
            samples.append((sample_dir.name, frames_folder))

    return samples


def get_frames_from_folder(frames_folder: Path, frame_stride: int = 1):
    frames = sorted([
        str(p)
        for p in frames_folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_FRAME_EXTENSIONS
    ])

    if frame_stride < 1:
        frame_stride = 1

    return frames[::frame_stride]


def get_frames_by_iframe_stride(frames_folder: Path, stride_seconds: int, margin: int = 3):
    frame_map = {}
    
    for p in frames_folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            match = re.search(r'(\d+)', p.name)
            if match:
                frame_num = int(match.group(1))
                frame_map[frame_num] = str(p)

    if not frame_map:
        return []

    available_frame_nums = sorted(frame_map.keys())
    max_frame = available_frame_nums[-1]
    
    selected_frames = []
    
    if available_frame_nums:
        selected_frames.append(frame_map[available_frame_nums[0]])

    target_frame = stride_seconds 
    
    while target_frame <= max_frame:
        best_frame_path = None
        
        if target_frame in frame_map:
            best_frame_path = frame_map[target_frame]
            
        else:
            min_dist = float('inf')
            for offset in range(-margin, margin + 1):
                if offset == 0:
                    continue
                
                candidate_num = target_frame + offset
                
                if candidate_num in frame_map:
                    dist = abs(offset)
                    if dist < min_dist:
                        min_dist = dist
                        best_frame_path = frame_map[candidate_num]

        if best_frame_path and best_frame_path != selected_frames[-1]:
            selected_frames.append(best_frame_path)
            
        target_frame += stride_seconds
        
    return selected_frames


def get_ground_truth(input_csv: str, sample_name: str):
    gt_epg = "Undefined"
    gt_manual = "Undefined"

    if not os.path.exists(input_csv):
        print(f"[AVISO] No existe CSV: {input_csv}")
        return gt_epg, gt_manual

    df = pd.read_csv(input_csv, skipinitialspace=True)

    if "ID" not in df.columns:
        raise ValueError("El CSV debe tener una columna llamada ID")

    matched_row = df[df["ID"].astype(str).str.strip() == sample_name]

    if matched_row.empty:
        print(f"[AVISO] No encontré etiquetas manuales/EPG en el CSV para {sample_name}")
        return gt_epg, gt_manual

    row_data = matched_row.iloc[0]

    gt_epg = str(row_data.get("CONTENT_CATEGORY_EPG", "Undefined")).strip()
    gt_manual = str(row_data.get("CONTENT_CATEGORY_MANUAL", "Undefined")).strip()

    if gt_epg == "" or gt_epg.lower() == "nan":
        gt_epg = "Undefined"

    if gt_manual == "" or gt_manual.lower() == "nan":
        gt_manual = "Undefined"

    return gt_epg, gt_manual


def compute_final_top3_votes(frame_predictions):
    votes = []

    for pred in frame_predictions:
        category = pred.get("prediction", "Undefined")
        if category != "Undefined":
            votes.append(category)

    total_votes = len(votes)

    if total_votes == 0:
        return []

    counter = Counter(votes)

    return [
        {
            "rank": idx + 1,
            "category": category,
            "votes": count,
            "total_votes": total_votes,
            "percentage": round((count / total_votes) * 100, 2),
        }
        for idx, (category, count) in enumerate(counter.most_common(3))
    ]


def save_video_json(
    frames_json_dir,
    sample_name,
    mode_name,
    resultado,
    gt_epg,
    gt_manual,
    coincide_epg,
    coincide_manual,
    frame_predictions,
    final_top3_votes,
    aggregate_metrics,
):
    frames_json_dir = Path(frames_json_dir)
    frames_json_dir.mkdir(parents=True, exist_ok=True)

    json_path = frames_json_dir / f"{sample_name}_{mode_name}.json"

    data = {
        "sample": sample_name,
        "mode": mode_name,
        "categoria_modelo": resultado,
        "categoria_epg": gt_epg,
        "categoria_manual": gt_manual,
        "match_epg_modelo": coincide_epg,
        "match_manual_modelo": coincide_manual,
        "num_frames": len(frame_predictions),
        "top_3_final_votes": final_top3_votes,
        "aggregate_metrics": aggregate_metrics,
        "predictions": frame_predictions,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"JSON por vídeo guardado en {json_path}")


def save_csv_row(
    output_csv,
    sample_name,
    mode_name,
    resultado,
    gt_epg,
    gt_manual,
    coincide_epg,
    coincide_manual,
    frame_predictions,
    final_top3_votes,
    aggregate_metrics,
):
    row = {
        "sample": sample_name,
        "mode": mode_name,
        "categoria_modelo": resultado,
        "categoria_epg": gt_epg,
        "categoria_manual": gt_manual,
        "match_epg_modelo": coincide_epg,
        "match_manual_modelo": coincide_manual,
        "num_frames": len(frame_predictions),

        "top1_final": final_top3_votes[0]["category"] if len(final_top3_votes) > 0 else "Undefined",
        "top1_votes": final_top3_votes[0]["votes"] if len(final_top3_votes) > 0 else 0,
        "top1_percentage": final_top3_votes[0]["percentage"] if len(final_top3_votes) > 0 else 0,

        "top2_final": final_top3_votes[1]["category"] if len(final_top3_votes) > 1 else "Undefined",
        "top2_votes": final_top3_votes[1]["votes"] if len(final_top3_votes) > 1 else 0,
        "top2_percentage": final_top3_votes[1]["percentage"] if len(final_top3_votes) > 1 else 0,

        "top3_final": final_top3_votes[2]["category"] if len(final_top3_votes) > 2 else "Undefined",
        "top3_votes": final_top3_votes[2]["votes"] if len(final_top3_votes) > 2 else 0,
        "top3_percentage": final_top3_votes[2]["percentage"] if len(final_top3_votes) > 2 else 0,

        "latency_total_ms": aggregate_metrics.get("latency_total_ms"),
        "latency_total_s": aggregate_metrics.get("latency_total_s"),
        "latency_per_token_s": aggregate_metrics.get("latency_per_token_s"),
        "tokens_per_second": aggregate_metrics.get("tokens_per_second"),

        "gpu_energy_j": aggregate_metrics.get("gpu_energy_j"),
        "cpu_energy_j": aggregate_metrics.get("cpu_energy_j"),
        "total_energy_j": aggregate_metrics.get("total_energy_j"),
        "energy_per_token_j": aggregate_metrics.get("energy_per_token_j"),

        "gpu_power_avg_w": aggregate_metrics.get("gpu_power_avg_w"),
        "cpu_power_avg_w": aggregate_metrics.get("cpu_power_avg_w"),
        "gpu_power_max_w": aggregate_metrics.get("gpu_power_max_w"),

        "gpu_util_avg_pct": aggregate_metrics.get("gpu_util_avg_pct"),
        "cpu_process_avg_pct": aggregate_metrics.get("cpu_process_avg_pct"),

        "gpu_mem_avg_mb": aggregate_metrics.get("gpu_mem_avg_mb"),
        "gpu_mem_max_mb": aggregate_metrics.get("gpu_mem_max_mb"),
        "ram_process_avg_mb": aggregate_metrics.get("ram_process_avg_mb"),

        "tokens_in_total": aggregate_metrics.get("tokens_in_total"),
        "tokens_out_total": aggregate_metrics.get("tokens_out_total"),
        "num_valid_frames": aggregate_metrics.get("num_valid_frames"),
    }

    if not os.path.exists(output_csv):
        pd.DataFrame([row]).to_csv(output_csv, index=False)
    else:
        pd.DataFrame([row]).to_csv(output_csv, mode="a", header=False, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frames-root-dir", default="./carpeta_frames_completa")
    parser.add_argument("--frames-subdir", default="frames_post_blur")
    parser.add_argument("--eit-dir", default="./eit")

    parser.add_argument("--input-csv", default="dataset_definitivo.csv")
    parser.add_argument("--output-csv", default="resumen_resultados.csv")
    parser.add_argument("--global-json", default="resultados_globales.json")
    parser.add_argument("--frames-json-dir", default="./resultados_frames")
    parser.add_argument("--output-dir", default="./resultados_frames") ###### <--- CAMBIO


    parser.add_argument("--text-only", type=str2bool, default=False)
    parser.add_argument("--frames-only", type=str2bool, default=False)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--seconds-stride", type=int, default=None)

    parser.add_argument("--two-stage", type=str2bool, default=False)
    parser.add_argument("--two-stage-mode", type=str, default="visual_then_epg", choices=["visual_then_epg", "epg_then_visual"])

    parser.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()
    
    if args.text_only and args.frames_only:
        print("ERROR: Las banderas --text-only y --frames-only son mutuamente exclusivas.")
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_root_dir = Path(args.frames_root_dir).resolve()
    eit_dir = Path(args.eit_dir).resolve()

    output_csv = output_dir / "resumen_resultados.csv"
    global_json = output_dir / "resultados_globales.json"
    frames_json_dir = output_dir / "resultados_frames"

    if not frames_root_dir.exists():
        print(f"ERROR: No existe la carpeta de frames: {frames_root_dir}")
        return

    if not eit_dir.exists():
        print(f"ERROR: No existe la carpeta EIT: {eit_dir}")
        return

    samples = find_frame_folders(frames_root_dir, args.frames_subdir)

    if not samples:
        print(f"No se encontraron carpetas con {args.frames_subdir}")
        return

    print(f"Se han encontrado {len(samples)} carpetas de frames.\n")

    print("Cargando modelo UNA sola vez...")
    classifier = VideoClassifier()
    print("Modelo cargado. Comienza el procesamiento batch.\n")

    all_results = []

    ok_count = 0
    fail_count = 0

    for idx, (sample_name, frames_folder) in enumerate(samples, start=1):
        eit_xml_path = eit_dir / f"{sample_name}.eit.xml"

        print("=" * 80)
        print(f"[{idx}/{len(samples)}] Procesando: {sample_name}")
        print(f"Frames: {frames_folder}")
        if not args.frames_only:
            print(f"EIT: {eit_xml_path}")
        else:
            print("EIT: Ignorada (Modo Visual Puro)")
        print("=" * 80)

        if not eit_xml_path.exists() and not args.frames_only:
            print(f"ERROR: No existe la EIT esperada: {eit_xml_path}")
            fail_count += 1
            if args.stop_on_error:
                break
            continue

        try:
            gt_epg, gt_manual = get_ground_truth(
                input_csv=args.input_csv,
                sample_name=sample_name,
            )

            if args.text_only:
                frames = []
                mode_name = "text_only"
                print("Modo TEXT_ONLY activado: solo se usará la EIT.")
            else:
                if args.seconds_stride is not None:
                    frames = get_frames_by_iframe_stride(
                        frames_folder=frames_folder,
                        stride_seconds=args.seconds_stride,
                        margin=3
                    )
                    mode_name = f"seconds_stride_{args.seconds_stride}"
                    if args.frames_only:
                        mode_name = f"frames_only_sec_stride_{args.seconds_stride}"
                    print(f"Muestreo temporal: 1 frame cada {args.seconds_stride}s")
                else:
                    frames = get_frames_from_folder(
                        frames_folder=frames_folder,
                        frame_stride=args.frame_stride,
                    )
                    mode_name = f"frames_stride_{args.frame_stride}"
                    if args.frames_only:
                        mode_name = f"frames_only_stride_{args.frame_stride}"

                print(f"Total frames encontrados/seleccionados: {len(frames)}")

            log.info("=" * 80)
            log.info(f"INICIANDO MUESTRA: {sample_name}")
            log.info(f"CARPETA FRAMES: {frames_folder}")
            log.info(f"EIT XML: {eit_xml_path if not args.frames_only else 'N/A'}")
            log.info(f"MODO: {mode_name}")
            log.info(f"N_FRAMES: {len(frames)}")
            log.info("=" * 80)

            resultado, frame_predictions, aggregate_metrics = classifier.classify(
                eit_xml_path=str(eit_xml_path) if not args.frames_only else None,
                image_paths=frames,
                text_only=args.text_only,
                frames_only=args.frames_only,
                two_stage=args.two_stage,
                two_stage_mode=args.two_stage_mode,
                sample_name=sample_name,
            )

            categoria_final = resultado.get("prediction", "Undefined")
            final_top3_votes = compute_final_top3_votes(frame_predictions)

            res_lower = categoria_final.strip().lower()
            coincide_epg = res_lower == gt_epg.lower() if gt_epg != "Undefined" else False
            coincide_manual = res_lower == gt_manual.lower() if gt_manual != "Undefined" else False

            print()
            print(f"MUESTRA: {sample_name}")
            print(f"CATEGORÍA MODELO: {categoria_final}")
            print(f"CATEGORÍA EPG: {gt_epg}")
            print(f"CATEGORÍA MANUAL: {gt_manual}")
            print(f"MODELO vs EPG: {'sí✅' if coincide_epg else 'no❌'}")
            print(f"MODELO vs MANUAL: {'sí✅' if coincide_manual else 'no❌'}")

            print("TOP 3 FINAL POR VOTOS:")
            for item in final_top3_votes:
                print(
                    f"{item['rank']}. {item['category']} "
                    f"({item['votes']}/{item['total_votes']} votos, {item['percentage']}%)"
                )

            print("MÉTRICAS AGREGADAS:")
            print(f"  Latencia total: {aggregate_metrics.get('latency_total_s')} s")
            print(f"  Latencia/token: {aggregate_metrics.get('latency_per_token_s')} s/token")
            print(f"  Tokens/s: {aggregate_metrics.get('tokens_per_second')} tok/s")
            print(f"  GPU energy: {aggregate_metrics.get('gpu_energy_j')} J")
            print(f"  CPU energy: {aggregate_metrics.get('cpu_energy_j')} J")
            print(f"  Total energy: {aggregate_metrics.get('total_energy_j')} J")
            print(f"  Energy/token: {aggregate_metrics.get('energy_per_token_j')} J/token")
            print(f"  GPU power avg: {aggregate_metrics.get('gpu_power_avg_w')} W")
            print(f"  CPU power avg: {aggregate_metrics.get('cpu_power_avg_w')} W")

            save_video_json(
                frames_json_dir=frames_json_dir,
                sample_name=sample_name,
                mode_name=mode_name,
                resultado=categoria_final,
                gt_epg=gt_epg,
                gt_manual=gt_manual,
                coincide_epg=coincide_epg,
                coincide_manual=coincide_manual,
                frame_predictions=frame_predictions,
                final_top3_votes=final_top3_votes,
                aggregate_metrics=aggregate_metrics,
            )

            save_csv_row(
                output_csv=output_csv,
                sample_name=sample_name,
                mode_name=mode_name,
                resultado=categoria_final,
                gt_epg=gt_epg,
                gt_manual=gt_manual,
                coincide_epg=coincide_epg,
                coincide_manual=coincide_manual,
                frame_predictions=frame_predictions,
                final_top3_votes=final_top3_votes,
                aggregate_metrics=aggregate_metrics,
            )

            global_result = {
                "sample": sample_name,
                "mode": mode_name,
                "categoria_modelo": categoria_final,
                "categoria_epg": gt_epg,
                "categoria_manual": gt_manual,
                "match_epg_modelo": coincide_epg,
                "match_manual_modelo": coincide_manual,
                "num_frames": len(frame_predictions),
                "top_3_final_votes": final_top3_votes,
                "aggregate_metrics": aggregate_metrics,
            }

            all_results.append(global_result)
            ok_count += 1

        except Exception as e:
            fail_count += 1
            print(f"ERROR procesando {sample_name}: {e}")

            if args.stop_on_error:
                break

    with open(global_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\n" + "#" * 80)
    print("EJECUCIÓN TERMINADA")
    print(f"Correctos: {ok_count}")
    print(f"Fallidos: {fail_count}")
    print(f"JSON global guardado en: {global_json}")
    print("#" * 80)


if __name__ == "__main__":
    main()