# TEVR ASR Tool

* state-of-the-art performance 
  * 3.64% WER on Common Voice German
  * rank #1 on [paperswithcode.com](https://paperswithcode.com/sota/speech-recognition-on-common-voice-german)
* no GPU needed
* 100% offline
* 100% private
* 100% free
* MIT license
* Linux x86_64
* command-line tool
* easy to understand
  * only 284 lines of C++ code
  * AI model on HuggingFace

## High Transcription Quality

In August 2022, we ranked 
**#1 on "Speech Recognition on Common Voice German (using extra training data)"**
with a 3.64% word error rate.
Accordingly, the performance of this tool is considered to be
the best of what's currently possible
in German speech recognition:
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tevr-improving-speech-recognition-by-token/speech-recognition-on-common-voice-german)](https://paperswithcode.com/sota/speech-recognition-on-common-voice-german?p=tevr-improving-speech-recognition-by-token)

## How does this work?

[L175-L185](https://github.com/DeutscheKI/tevr-asr-tool/blob/v1.0.0/tevr_asr_tool.cc#L175-L185) 
load the WAV file.
[L189-L229](https://github.com/DeutscheKI/tevr-asr-tool/blob/v1.0.0/tevr_asr_tool.cc#L189-L229)
execute the acoustic AI model.
[L260-L275](https://github.com/DeutscheKI/tevr-asr-tool/blob/v1.0.0/tevr_asr_tool.cc#L260-L275)
convert the predicted token logits into string snippets.
[L73-L162](https://github.com/DeutscheKI/tevr-asr-tool/blob/v1.0.0/tevr_asr_tool.cc#L73-L162)
implement the Beam search re-scoring based on a KenLM language model. 

If you're curious how the acoustic AI model works 
and why I designed it that way, here's the paper:
https://arxiv.org/abs/2206.12693
and here's a pre-trained HuggingFace transformers model:
https://huggingface.co/fxtentacle/wav2vec2-xls-r-1b-tevr


## Install the Debian/Ubuntu package
Download `tevr_asr_tool-1.0.0-Linux-x86_64.deb` from GitHub
and extract the multipart ZIP:
```bash
wget "https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.001"
wget "https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.002"
wget "https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.003"
wget "https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.004"
wget "https://github.com/DeutscheKI/tevr-asr-tool/releases/download/v1.0.0/tevr_asr_tool-1.0.0-Linux-x86_64.zip.005"
cat tevr_asr_tool-1.0.0-Linux-x86_64.zip.00* > tevr_asr_tool-1.0.0-Linux-x86_64.zip
unzip tevr_asr_tool-1.0.0-Linux-x86_64.zip
```
Install it:
```bash
sudo dpkg -i tevr_asr_tool-1.0.0-Linux-x86_64.deb
```

## Install from Source Code
Download submodules:
```bash
git submodule update --init
```
CMake configure and build:
```bash
cmake -DCMAKE_BUILD_TYPE=MinSizeRel -DCPACK_CMAKE_GENERATOR=Ninja -S . -B build
cmake --build build --target tevr_asr_tool -j 16
```
Create debian package:
```bash
(cd build && cpack -G DEB)
```
Install it:
```bash
sudo dpkg -i build/tevr_asr_tool-1.0.0-Linux-x86_64.deb
```

## Usage

```bash
tevr_asr_tool --target_file=test_audio.wav 2>log.txt
```
should display the correct transcription
` mückenstiche sollte man nicht aufkratzen `.
And `log.txt` will contain the diagnostics and progress 
that was logged to stderr during execution.

## GPU Acceleration for Developers

I plan to release a Vulkan & OpenGL-accelerated 
real-time low-latency transcription 
software for developers soon.
It'll run 100% private + 100% offline 
just like this tool,
but instead of processing a WAV file on CPU
it'll stream the real-time GPU transcription 
of your microphone input
through a WebRTC-capable REST API
so that you can easily integrate it 
with your own voice-controlled projects.
For example, that'll enable 
hackable voice typing 
together with `pynput.keyboard`.

If you want to get notified when it launches,
please enter your email at
https://madmimi.com/signups/f0da3b13840d40ce9e061cafea6280d5/join

## Commercial Customization

This tool itself is free to use also for commercial use.
And of course it comes with no warranty of any kind.

But if you have an idea for a commercial use-case for 
a customized version of this tool or for similar 
technology - ideally something that helps 
small and medium-sized businesses in northern Germany
become more competitive - 
then please contact me at moin@DeutscheKI.de



## Research Citation

If you use this for research, please cite:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.12693,
  doi = {10.48550/ARXIV.2206.12693},
  url = {https://arxiv.org/abs/2206.12693},
  author = {Krabbenhöft, Hajo Nils and Barth, Erhardt},  
  keywords = {Computation and Language (cs.CL), Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, F.2.1; I.2.6; I.2.7},  
  title = {TEVR: Improving Speech Recognition by Token Entropy Variance Reduction},  
  publisher = {arXiv},  
  year = {2022}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Replace the AI Model

The German AI model and my training scripts can be found on HuggingFace:
https://huggingface.co/fxtentacle/wav2vec2-xls-r-1b-tevr

The model has undergone XLS-R cross-language pre-training.
You can directly fine-tune it with a different 
language dataset - for example CommonVoice English - 
and then re-export the files in the
`tevr-asr-data` folder.

Alternatively, you can donate roughly 2 weeks of 
A100 GPU credits to me 
and I'll train a suitable recognition model
and upload it to HuggingFace.
