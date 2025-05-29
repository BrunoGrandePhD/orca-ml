#!/usr/bin/env python3

import argparse
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import requests
import torch
import torchaudio
from tqdm import tqdm


class AudioProcessingError(Exception):
    """Base class for audio processing errors."""


class InvalidJSONError(AudioProcessingError):
    """Invalid JSON input file."""


class AudioDownloadError(AudioProcessingError):
    """Error downloading audio file."""


class AudioValidationError(AudioProcessingError):
    """Invalid audio file."""


class OutputFileExistsError(AudioProcessingError):
    """Output file already exists."""


@dataclass
class Annotation:
    id: int
    start_time: float
    end_time: float
    confidence: float


@dataclass
class AudioFile:
    id: str
    audio_uri: str
    found: str
    annotations: List[Annotation]

    @classmethod
    def from_dict(cls, data: dict) -> "AudioFile":
        """Create AudioFile from dictionary."""
        annotations = [
            Annotation(
                id=ann["id"],
                start_time=float(ann["startTime"]),
                end_time=float(ann["endTime"]),
                confidence=float(ann["confidence"]),
            )
            for ann in data.get("annotations", [])
        ]

        return cls(
            id=data["id"],
            audio_uri=data["audioUri"],
            found=data.get("found", "unknown"),
            annotations=annotations,
        )


def load_audio_files(json_path: str, shuffle: bool = True) -> List[AudioFile]:
    """
    Load and validate audio files from JSON.

    Args:
        json_path: Path to JSON file containing audio file descriptions
        shuffle: Whether to randomly shuffle the audio files (default: True)
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise InvalidJSONError("JSON file must contain a list of audio files")

        audio_files = [AudioFile.from_dict(item) for item in data]
        if shuffle:
            random.shuffle(audio_files)
        return audio_files

    except json.JSONDecodeError as e:
        raise InvalidJSONError(f"Invalid JSON format: {str(e)}")
    except KeyError as e:
        raise InvalidJSONError(f"Missing required field: {str(e)}")
    except Exception as e:
        raise InvalidJSONError(f"Error loading JSON file: {str(e)}")


def validate_wav_header(buffer: io.BytesIO) -> None:
    """Validate WAV file header and format."""
    buffer.seek(0)

    # Check RIFF header
    if buffer.read(4) != b"RIFF":
        raise AudioValidationError("Invalid WAV file: Missing RIFF header")

    # Skip file size
    buffer.read(4)

    # Check WAVE format
    if buffer.read(4) != b"WAVE":
        raise AudioValidationError("Invalid WAV file: Not a WAVE file")

    # Find fmt chunk
    while True:
        chunk_id = buffer.read(4)
        if not chunk_id:
            raise AudioValidationError("Invalid WAV file: Missing format chunk")
        if chunk_id == b"fmt ":
            break
        # Skip other chunks
        chunk_size = int.from_bytes(buffer.read(4), "little")
        buffer.seek(chunk_size, 1)

    # Validate format
    fmt_size = int.from_bytes(buffer.read(4), "little")
    if fmt_size < 16:
        raise AudioValidationError("Invalid WAV file: Incomplete format chunk")

    # Check PCM format
    audio_format = int.from_bytes(buffer.read(2), "little")
    if audio_format != 1:  # 1 = PCM
        raise AudioValidationError("Unsupported WAV format: Only PCM format is supported")

    buffer.seek(0)


def download_audio(url: str, normalize: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Download audio file and return tensor and sample rate.

    Args:
        url: URL of the audio file to download
        normalize: Whether to normalize audio to [-1, 1] range (default: True)
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Read into memory buffer
        buffer = io.BytesIO()
        total_size = int(response.headers.get("content-length", 0))

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    buffer.write(chunk)
                    pbar.update(len(chunk))

        buffer.seek(0)

        # Validate WAV format
        validate_wav_header(buffer)

        # Load audio using torchaudio with explicit parameters
        waveform, sample_rate = torchaudio.load(
            buffer,
            format="wav",
            channels_first=True,
            normalize=normalize,
        )

        # Check for invalid values
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            raise AudioValidationError("Invalid audio data: Contains NaN or infinite values")

        return waveform, sample_rate

    except requests.RequestException as e:
        raise AudioDownloadError(f"Failed to download audio: {str(e)}")
    except Exception as e:
        raise AudioDownloadError(f"Error processing audio data: {str(e)}")


def convert_to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Convert stereo audio to mono using proper channel mixing and normalization."""
    if audio.shape[0] == 1:
        return audio

    # Average the channels
    mono = torch.mean(audio, dim=0, keepdim=True)

    # Remove DC offset
    mono = mono - torch.mean(mono)

    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(mono))
    if max_val > 0:
        mono = mono / max_val

    return mono


def validate_audio(audio: torch.Tensor, force_mono: bool = True) -> torch.Tensor:
    """
    Validate audio tensor format and properties, and optionally convert to mono.

    Args:
        audio: Input audio tensor
        force_mono: Whether to convert stereo audio to mono (default: True)
    """
    if not isinstance(audio, torch.Tensor):
        raise AudioValidationError("Audio must be a torch.Tensor")

    if len(audio.shape) != 2:
        raise AudioValidationError(
            f"Expected 2D tensor (channels, samples), got shape {audio.shape}"
        )

    if audio.shape[0] not in [1, 2]:
        raise AudioValidationError(f"Expected 1 or 2 channels, got {audio.shape[0]}")

    if torch.isnan(audio).any():
        raise AudioValidationError("Audio contains NaN values")

    # Convert to mono if stereo and force_mono is True
    return convert_to_mono(audio) if force_mono else audio


def apply_crossfade(
    audio1: torch.Tensor, audio2: torch.Tensor, fade_duration: float = 0.1
) -> torch.Tensor:
    """Apply crossfade between two audio segments."""
    # Convert fade duration from seconds to samples (assuming 44.1kHz)
    fade_samples = int(44100 * fade_duration)

    if audio1.shape[1] < fade_samples * 2 or audio2.shape[1] < fade_samples * 2:
        # If segments are too short, just concatenate
        return torch.cat([audio1, audio2], dim=1)

    # Create fade curves
    fade_out = torch.linspace(1, 0, fade_samples)
    fade_in = torch.linspace(0, 1, fade_samples)

    # Apply fade out to the end of first segment
    audio1_end = audio1[:, -fade_samples:] * fade_out
    # Apply fade in to the start of second segment
    audio2_start = audio2[:, :fade_samples] * fade_in

    # Combine the faded portions
    crossfade = audio1_end + audio2_start

    # Concatenate everything
    result = torch.cat([audio1[:, :-fade_samples], crossfade, audio2[:, fade_samples:]], dim=1)

    return result


def stitch_audio_files(
    audio_files: List[AudioFile],
    progress_callback: Optional[Callable] = None,
    force_mono: bool = True,
    normalize: bool = True,
    enable_resampling: bool = True,
    enable_crossfade: bool = True,
    target_sample_rate: int = 48000,
) -> Tuple[torch.Tensor, List[float], int]:
    """
    Stitch audio files and return:
    - Combined audio tensor
    - List of boundary positions in seconds
    - Sample rate of the combined audio

    Args:
        audio_files: List of audio files to process
        progress_callback: Optional callback for progress updates
        force_mono: Whether to convert audio to mono (default: True)
        normalize: Whether to normalize audio (default: True)
        enable_resampling: Whether to resample audio to target rate (default: True)
        enable_crossfade: Whether to apply crossfade between segments (default: True)
        target_sample_rate: Target sample rate for resampling (default: 48000)
    """
    if not audio_files:
        raise AudioProcessingError("No audio files provided")

    # Process first file
    first_audio, sample_rate = download_audio(audio_files[0].audio_uri, normalize=normalize)
    print(f"[DEBUG] File {audio_files[0].id}: Original sample rate = {sample_rate} Hz")
    first_audio = validate_audio(first_audio, force_mono=force_mono)

    # Resample first file if needed and enabled
    if enable_resampling and sample_rate != target_sample_rate:
        print(
            f"[DEBUG] Resampling {audio_files[0].id} from {sample_rate} Hz to {target_sample_rate} Hz"
        )
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
            lowpass_filter_width=64,  # Higher quality resampling
            rolloff=0.9475,  # Sharper filter
            resampling_method="sinc_interp_hann",
            beta=14.769656459379492,  # Kaiser window parameter
        )
        first_audio = resampler(first_audio)
        sample_rate = target_sample_rate

    combined_audio = first_audio
    current_length = first_audio.shape[1]
    boundaries = [0.0, current_length / sample_rate]

    # Process remaining files
    for i, audio_file in enumerate(audio_files[1:], start=1):
        if progress_callback:
            progress_callback(i, len(audio_files))

        audio, curr_sample_rate = download_audio(audio_file.audio_uri, normalize=normalize)
        print(f"[DEBUG] File {audio_file.id}: Original sample rate = {curr_sample_rate} Hz")
        audio = validate_audio(audio, force_mono=force_mono)

        # Resample if needed and enabled
        if enable_resampling and curr_sample_rate != target_sample_rate:
            print(
                f"[DEBUG] Resampling {audio_file.id} from {curr_sample_rate} Hz to {target_sample_rate} Hz"
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=curr_sample_rate,
                new_freq=target_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475,
                resampling_method="sinc_interp_hann",
                beta=14.769656459379492,
            )
            audio = resampler(audio)

        # Apply crossfade when combining segments if enabled
        if enable_crossfade:
            combined_audio = apply_crossfade(combined_audio, audio)
        else:
            combined_audio = torch.cat([combined_audio, audio], dim=1)

        current_length = combined_audio.shape[1]
        boundaries.append(current_length / sample_rate)

    return combined_audio, boundaries, sample_rate


def generate_annotation_labels(audio_files: List[AudioFile], boundaries: List[float]) -> List[str]:
    """Generate Audacity label file content for annotations."""
    labels = []

    for i, audio_file in enumerate(audio_files):
        offset = boundaries[i]
        label_type = (
            "tp" if audio_file.found == "yes" else "fp" if audio_file.found == "no" else "unknown"
        )

        for ann in audio_file.annotations:
            start = offset + ann.start_time
            end = offset + ann.end_time
            labels.append(f"{start:.6f}\t{end:.6f}\t{label_type}")

    return labels


def generate_boundary_labels(audio_files: List[AudioFile], boundaries: List[float]) -> List[str]:
    """Generate Audacity label file content for file boundaries."""
    return [
        f"{start:.6f}\t{end:.6f}\t{audio_file.id}"
        for start, end, audio_file in zip(boundaries[:-1], boundaries[1:], audio_files)
    ]


def save_audio(audio: torch.Tensor, path: str, sample_rate: int = 44100) -> None:
    """Save audio tensor to WAV file."""
    try:
        torchaudio.save(path, audio, sample_rate, format="wav")
    except Exception as e:
        raise AudioProcessingError(f"Failed to save audio file: {str(e)}")


def save_labels(labels: List[str], path: str) -> None:
    """Save labels to Audacity label file."""
    try:
        with open(path, "w") as f:
            for label in labels:
                f.write(f"{label}\n")
    except Exception as e:
        raise AudioProcessingError(f"Failed to save labels file: {str(e)}")


def process_audio_files(
    input_path: str,
    output_prefix: str,
    overwrite: bool = False,
    shuffle: bool = True,
    limit: Optional[int] = None,
    force_mono: bool = True,
    normalize: bool = True,
    enable_resampling: bool = True,
    enable_crossfade: bool = True,
    target_sample_rate: int = 48000,
) -> Tuple[str, str, str]:
    """
    Main processing function.

    Args:
        input_path: Path to input JSON file
        output_prefix: Prefix for output files
        overwrite: Whether to overwrite existing files
        shuffle: Whether to shuffle audio files
        limit: Maximum number of audio files to process
        force_mono: Whether to convert audio to mono (default: True)
        normalize: Whether to normalize audio (default: True)
        enable_resampling: Whether to resample audio to target rate (default: True)
        enable_crossfade: Whether to apply crossfade between segments (default: True)
        target_sample_rate: Target sample rate for resampling (default: 48000)
    """
    # Prepare output paths
    output_paths = {
        "audio": f"{output_prefix}_stitched.wav",
        "annotations": f"{output_prefix}_annotations.txt",
        "boundaries": f"{output_prefix}_boundaries.txt",
    }

    # Check if files exist
    if not overwrite:
        existing = [path for path in output_paths.values() if Path(path).exists()]
        if existing:
            raise OutputFileExistsError(
                f"Output files already exist: {', '.join(existing)}. Use --overwrite to force."
            )

    # Load and process files
    audio_files = load_audio_files(input_path, shuffle=shuffle)
    if limit is not None:
        audio_files = audio_files[:limit]

    with tqdm(total=len(audio_files), desc="Processing audio files") as pbar:
        combined_audio, boundaries, sample_rate = stitch_audio_files(
            audio_files,
            progress_callback=lambda i, total: pbar.update(1),
            force_mono=force_mono,
            normalize=normalize,
            enable_resampling=enable_resampling,
            enable_crossfade=enable_crossfade,
            target_sample_rate=target_sample_rate,
        )

    # Generate and save outputs
    save_audio(combined_audio, output_paths["audio"], sample_rate)

    annotation_labels = generate_annotation_labels(audio_files, boundaries)
    save_labels(annotation_labels, output_paths["annotations"])

    boundary_labels = generate_boundary_labels(audio_files, boundaries)
    save_labels(boundary_labels, output_paths["boundaries"])

    return output_paths["audio"], output_paths["annotations"], output_paths["boundaries"]


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Stitch audio files and generate Audacity labels")
    parser.add_argument("input_json", help="Input JSON file containing audio file descriptions")
    parser.add_argument("output_prefix", help="Prefix for output files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument(
        "--no-shuffle", action="store_true", help="Disable shuffling of audio files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of audio files to process",
        metavar="N",
    )
    parser.add_argument(
        "--no-mono",
        action="store_true",
        help="Keep stereo audio instead of converting to mono",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable audio normalization",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Disable resampling to target sample rate",
    )
    parser.add_argument(
        "--no-crossfade",
        action="store_true",
        help="Disable crossfading between audio segments",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Target sample rate in Hz (default: 48000)",
    )

    args = parser.parse_args()

    try:
        process_audio_files(
            args.input_json,
            args.output_prefix,
            overwrite=args.overwrite,
            shuffle=not args.no_shuffle,
            limit=args.limit,
            force_mono=not args.no_mono,
            normalize=not args.no_normalize,
            enable_resampling=not args.no_resample,
            enable_crossfade=not args.no_crossfade,
            target_sample_rate=args.sample_rate,
        )
    except AudioProcessingError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
