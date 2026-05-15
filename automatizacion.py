import subprocess
import sys

def main():
    experimentos = [
        {
            "nombre": "Text_Only",
            "argumentos": [
                "--output-dir", "resultados/experimento_text_only",
                "--text-only", "True"
            ]
        },
        {
            "nombre": "Frames_Only_30s",
            "argumentos": [
                "--output-dir", "resultados/experimento_frames_only_30s",
                "--frames-only", "True",
                "--seconds-stride", "30"
            ]
        },
                {
            "nombre": "Frames_Only_60s",
            "argumentos": [
                "--output-dir", "resultados/experimento_frames_only_60s",
                "--frames-only", "True",
                "--seconds-stride", "60"
            ]
        },
        {
            "nombre": "Multimodal_30s",
            "argumentos": [
                "--output-dir", "resultados/experimento_multimodal_30s",
                "--text-only", "false",
                "--frames-only", "false",
                "--seconds-stride", "30"
            ]
        },
        {
            "nombre": "Multimodal_60s",
            "argumentos": [
                "--output-dir", "resultados/experimento_multimodal_60s",
                "--text-only", "false",
                "--frames-only", "false",
                "--seconds-stride", "60"
            ]
        },
        {
            "nombre": "Visual_Then_EPG",
            "argumentos": [
                "--output-dir", "resultados/experimento_visual_then_epg",
                "--two-stage", "True",
                "--two-stage-mode", "visual_then_epg",
                "--seconds-stride", "30"
            ]
        },
        {
            "nombre": "EPG_Then_Visual",
            "argumentos": [
                "--output-dir", "resultados/experimento_epg_then_visual",
                "--two-stage", "True",
                "--two-stage-mode", "epg_then_visual",
                "--seconds-stride", "30"
            ]
        }
    ]

    base_script = "run_all_videos_docker.py"

    print("=" * 50)
    print("INICIANDO AUTOMATIZACIÓN DE EXPERIMENTOS")
    print("=" * 50)

    for exp in experimentos:
        print(f"\n Lanzamiento experimento: {exp['nombre']}")
        
        comando = [sys.executable, base_script] + exp["argumentos"]
        
        try:
            subprocess.run(comando, check=True)
            print(f"Experimento {exp['nombre']} completado.")
            
        except subprocess.CalledProcessError as e:
            print(f"Fallo en {exp['nombre']}.")
            break 


if __name__ == "__main__":
    main()
