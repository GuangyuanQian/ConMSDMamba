from funasr import AutoModel

model_id = "iic/emotion2vec_base"
model = AutoModel(
    model=model_id,
    hub="ms",
)

wav_file = f"/IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"  # example
rec_result = model.generate(wav_file, output_dir="./outputs", granularity="frame")
print(rec_result)
