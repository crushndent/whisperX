import os
import shutil
import subprocess
import time
import argparse
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.mp4', '.flac', '.ogg'}

def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(Path(root) / file)
    return audio_files

def transcribe_file(source_file, input_dir):
    filename = source_file.name
    destination_mp3 = input_dir / filename
    
    # 1. Copy audio file to input directory
    print(f"--> Copying {filename} to container input...")
    shutil.copy2(source_file, destination_mp3)

    # 2. Run transcribe command
    # Path inside container is always /data/[filename]
    container_path = f"/data/{filename}"
    cmd = ["docker", "compose", "run", "--rm", "whisperx", container_path]
    
    print(f"--> Starting transcription for {filename}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error transcribing {filename}: {e}")
        # Clean up mp3 even on failure
        if destination_mp3.exists():
            os.remove(destination_mp3)
        return False

    # 3. Check for completion and correct output filenames
    transcript_filename = source_file.stem + "_transcript.txt"
    generated_transcript = input_dir / transcript_filename
    
    return_filename = source_file.stem + "_return.txt"
    generated_return = input_dir / return_filename

    if generated_transcript.exists():
        print("--> Transcription complete. Waiting 5 seconds...")
        time.sleep(5)
        
        # 4. Move new files back to original location
        original_location_transcript = source_file.parent / transcript_filename
        print(f"--> Moving transcript to {original_location_transcript}")
        shutil.move(str(generated_transcript), str(original_location_transcript))
        
        if generated_return.exists():
            original_location_return = source_file.parent / return_filename
            print(f"--> Moving return file to {original_location_return}")
            shutil.move(str(generated_return), str(original_location_return))
        else:
            print("!!! Warning: Raw return file not found.")

        # 5. Delete the copied mp3 from input directory
        print("--> Cleaning up input file...")
        if destination_mp3.exists():
            os.remove(destination_mp3)
            
        print(f"SUCCESS: Finished {filename}\n")
        return True
    else:
        print(f"!!! Error: Expected transcript file {generated_transcript} not found.\n")
        # Clean up mp3
        if destination_mp3.exists():
            os.remove(destination_mp3)
        return False

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files using WhisperX Docker.")
    parser.add_argument("source_dir", help="Directory to search for audio files")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist.")
        sys.exit(1)

    # The 'input' dir is in the same directory as this script
    script_dir = Path(__file__).parent.resolve()
    input_dir = script_dir / "input"
    
    if not input_dir.exists():
        print(f"Creating input directory at {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for audio files in {source_dir}...\n")
    audio_files = get_audio_files(source_dir)
    
    if not audio_files:
        print("No audio files found.")
        return

    print(f"Found {len(audio_files)} files to process.")
    
    success_count = 0
    for file_path in audio_files:
        if transcribe_file(file_path, input_dir):
            success_count += 1
            
    print("="*40)
    print(f"Batch processing complete. {success_count}/{len(audio_files)} files transcribed.")
    print("="*40)

if __name__ == "__main__":
    main()
