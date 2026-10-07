General-speech check used by `better-voice-train`: 20 short clips of other people reading.
Training is only applied if the model doesn't get worse on these, so it can't learn your
voice by forgetting how to hear everyone else.

Audio and text: LibriSpeech ASR corpus (dev-clean sample, via hf-internal-testing/librispeech_asr_dummy),
V. Panayotov, G. Chen, D. Povey, S. Khudanpur, licensed CC BY 4.0
(https://www.openslr.org/12).
