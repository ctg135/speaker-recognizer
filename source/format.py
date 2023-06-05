from pydub import AudioSegment
import os

def format_mp3s(input_dir, output_dir, output_format):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.{output_format}")

            # Load the MP3 file using pydub
            audio = AudioSegment.from_mp3(input_path)

            # Convert to single channel (mono)
            audio = audio.set_channels(1)

            # Resample to 44100 sample rate
            audio = audio.set_frame_rate(44100)
            # Export the audio in the desired format
            audio.export(output_path, format=output_format)
            print(f"Formatted {filename} to {output_format}")

# Example usage
input_directory = "samples"
output_directory = "sample_format"
output_format = "wav"
print("dsadasdas")
format_mp3s(input_directory, output_directory, output_format)
