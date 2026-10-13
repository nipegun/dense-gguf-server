# dense-gguf-server

यह README इन भाषाओं में भी उपलब्ध है:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), [he-IL](README.he-IL.md), **hi-IN**, [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## यह क्या है

वेब चैट और HTTP इन्फ़रेंस API के माध्यम से सघन GGUF मॉडल स्थानीय रूप से चलाएँ।

llama.cpp जैसा, लेकिन अधिकतम संभव इन्फ़रेंस गति पाने के लिए निपेगुनियन स्कैल्पेल की सटीकता से तराशा गया।

## स्क्रीनशॉट

अभी स्क्रीनशॉट नहीं जोड़े गए हैं।

## परिनियोजन

**Debian** पर परियोजना को बिल्ड करने के बाद, उसके मूल फ़ोल्डर से `root` के रूप में ये कमांड चलाएँ:

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

`http://127.0.0.1:8080` खोलें। API का पथ `/api`, OpenAI-संगत क्लाइंट का पथ `/api/v1` और Swagger का पथ `/api/doc/` है। मॉडल वेट अलग से उपलब्ध कराने होंगे; दूसरे क्लाइंट को प्रवेश देते समय `--api-key` का उपयोग करें।

## बिल्ड

पहले Debian 13 पर `root` के रूप में निर्भरताएँ स्थापित करें:

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

फिर अपने सामान्य उपयोगकर्ता खाते से परियोजना फ़ोल्डर में चलाएँ:

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

स्क्रिप्ट स्थानीय स्रोतों को `_/build/` में बिल्ड करती है। `DGS_BUILD_JOBS=8` समानांतर कार्यों की संख्या तय करता है। अतिरिक्त तर्क CMake को दिए जाते हैं, जैसे संबंधित SDK स्थापित होने पर `-DGGML_CUDA=ON` या `-DGGML_VULKAN=ON`। [बिल्ड निर्भरताएँ और विकल्प](MANUAL.hi-IN.md#local-build) पहले से उपलब्ध होने चाहिए।

## दस्तावेज़ पढ़ें

ये दस्तावेज़ देखें:

- [doc/MANUAL.hi-IN.md](MANUAL.hi-IN.md): DGS बिल्ड, स्थापना और उपयोग, MCP तथा वेब सेटिंग्स और समस्याओं की जाँच।
- [doc/CODE.hi-IN.md](CODE.hi-IN.md): सघन इंजन, स्रोत संरचना और बैकएंड/फ़्रंटएंड के बिल्ड अनुबंध।

## परियोजना का समर्थन करें

- नई सुविधाएँ जोड़ने के लिए **[मुझे काम सौंपें](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)**।
- **GitHub Sponsors।** आप GitHub पर परियोजना को आर्थिक सहायता भी दे सकते हैं: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun)।
