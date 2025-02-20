import subprocess
import os

if __name__ == "__main__":
    output_dir = 'outs/beit_linear_probe/renders'
    for folder in os.listdir(output_dir):
        subprocess.run(f'ffmpeg -framerate 10 -pattern_type glob -i \'{output_dir}/{folder}/hmp/*.png\' -c:v libx264 -preset veryslow -crf 24 {output_dir}/{folder}/hmp/{folder}.mp4', shell=True)