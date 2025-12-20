#!/usr/bin/env python3
"""
WhisperX Transcription with Speaker Diarization
Usage: python transcribe_with_diarization.py <audio_file>
"""
import os
import sys
from dotenv import load_dotenv
import whisperx
from whisperx.diarize import DiarizationPipeline

# Load environment variables from .env file
load_dotenv()

def transcribe_with_speakers(audio_file, model_size="large-v2", device=None, compute_type="int8"):
    """
    Transcribe audio file with speaker diarization.

    Args:
        audio_file: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
        device: "cpu" or "cuda"
        compute_type: "int8", "float16", or "float32"
    """
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "your_token_here":
        raise ValueError(
            "Please set your HuggingFace token in the .env file!\n"
            "1. Edit .env file\n"
            "2. Replace 'your_token_here' with your actual token\n"
            "3. Get token from: https://huggingface.co/settings/tokens"
        )

    print(f"Loading audio file: {audio_file}")
    audio = whisperx.load_audio(audio_file)

    # 1. Transcribe with Whisper
    print(f"\n1. Transcribing with Whisper {model_size}...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=8)
    print(f"   Detected language: {result['language']}")

    # 2. Align whisper output
    print("\n2. Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )

    # 3. Assign speaker labels
    print("\n3. Performing speaker diarization...")
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Print results
    print("\n" + "="*80)
    print("TRANSCRIPTION WITH SPEAKER LABELS")
    print("="*80 + "\n")

    # Save raw (unmerged) results
    raw_output_file = audio_file.rsplit('.', 1)[0] + '_return.txt'
    with open(raw_output_file, 'w', encoding='utf-8') as f:
        for segment in result["segments"]:
            speaker = segment.get('speaker', 'UNKNOWN')
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            f.write(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n")
    print(f"\nRaw return saved to: {raw_output_file}")

    # Merge segments by speaker
    merged_segments = []
    if result["segments"]:
        current_segment = result["segments"][0]
        current_segment['speaker'] = current_segment.get('speaker', 'UNKNOWN')
        
        for next_segment in result["segments"][1:]:
            next_speaker = next_segment.get('speaker', 'UNKNOWN')
             
            if next_speaker == current_segment['speaker']:
                # Same speaker, merge text
                current_segment['text'] += "\n" + next_segment['text'].strip()
                current_segment['end'] = next_segment['end']
            else:
                # Different speaker, push current and start new
                merged_segments.append(current_segment)
                current_segment = next_segment
                current_segment['speaker'] = next_speaker
        
        merged_segments.append(current_segment)

    # Print results
    print("\n" + "="*80)
    print("TRANSCRIPTION WITH SPEAKER LABELS")
    print("="*80 + "\n")

    for segment in merged_segments:
        speaker = segment['speaker']
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        print(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")

    # Save results
    output_file = audio_file.rsplit('.', 1)[0] + '_transcript.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in merged_segments:
            speaker = segment['speaker']
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            f.write(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}\n")

    print(f"\nTranscript saved to: {output_file}")

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_with_diarization.py <audio_file>")
        print("\nExample:")
        print("  python transcribe_with_diarization.py my_recording.mp3")
        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # Use auto-detection by default
    transcribe_with_speakers(audio_file, model_size="large-v2", device=None, compute_type="int8")
