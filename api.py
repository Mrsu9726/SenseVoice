# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from io import BytesIO


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"

# optional Qwen model for semantic correction
if os.getenv("USE_QWEN", "false").lower() == "true":
    qwen_name = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2-7B-Chat")
    qwen_model = AutoModelForCausalLM.from_pretrained(qwen_name, device_map="auto")
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_name)
else:
    qwen_model = None
    qwen_tokenizer = None

def qwen_correct(text: str) -> str:
    if qwen_model is None:
        return text
    prompt = f"请根据上下文还原成合理的完整句子:\n{text}\n还原后："
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
    outputs = qwen_model.generate(**inputs, max_new_tokens=128)
    result = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "还原后：" in result:
        result = result.split("还原后：", 1)[-1].strip()
    return result

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """

@app.post("/api/v1/asr")
async def turn_audio_to_text(files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")], keys: Annotated[str, Form(description="name of each audio joined with comma")], lang: Annotated[Language, Form(description="language of audio content")] = "auto"):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language=lang, # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
        it["corrected_text"] = qwen_correct(it["text"])
    return {"result": res[0]}
