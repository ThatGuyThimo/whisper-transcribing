import whisper
import torch
from pyannote.audio import Pipeline

lang = "nl"
selected_model = "medium"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Load Whisper model on GPU
model = whisper.load_model(selected_model).to(device)

# Load the speaker diarization pipeline from pyannote
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=""
)

# Apply speaker diarization
diarization = pipeline("audio/audio.mp3")

# Load audio for Whisper transcription
audio = whisper.load_audio("audio/audio.mp3")

# Initialize the full transcription with speaker labels
full_transcription = ""

# Process each speaker segment
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Extract the corresponding audio segment
    start, end = int(turn.start * 16000), int(turn.end * 16000)  # Convert to samples
    audio_segment = audio[start:end]

    # Check if the audio segment is longer than 30 seconds
    segment_duration = end - start
    chunk_size = 30 * 16000  # 30 seconds in samples

    # If the segment is longer than 30 seconds, split it into chunks
    if segment_duration > chunk_size:
        for i in range(0, segment_duration, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, segment_duration)
            audio_chunk = audio_segment[chunk_start:chunk_end]

            # Transcribe the audio chunk using Whisper
            audio_chunk = whisper.pad_or_trim(audio_chunk)
            mel = whisper.log_mel_spectrogram(audio_chunk).to(device)

            # Decode the audio
            options = whisper.DecodingOptions(language=lang, patience=2.0, beam_size=5)
            result = whisper.decode(model, mel, options)
            print(f"in {speaker}: {result.text}")

            # Add the transcribed text along with speaker information
            full_transcription += f"{speaker}: {result.text}\n"
    else:
        # Transcribe the full audio segment if it's shorter than 30 seconds
        audio_segment = whisper.pad_or_trim(audio_segment)
        mel = whisper.log_mel_spectrogram(audio_segment).to(device)

        # Decode the audio
        options = whisper.DecodingOptions(language=lang, patience=2.0, beam_size=5)
        result = whisper.decode(model, mel, options)
        print(f"out {speaker}: {result.text}")

        # Add the transcribed text along with speaker information
        full_transcription += f"{speaker}: {result.text}\n"

# Print or save the transcription
try:
    with open('speaker_diarization_output.txt', 'w', encoding='utf-8') as file:
        file.write(full_transcription.strip())
    print("Speaker diarization and transcription written to speaker_diarization_output.txt")
except Exception as e:
    print(f"Error writing to file: {e}")
