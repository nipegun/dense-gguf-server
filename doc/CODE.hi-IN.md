# DGS — कोड मानचित्र

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · [he-IL](CODE.he-IL.md) · **hi-IN** · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server एक स्वतंत्र परियोजना है, जिसे **NiPeGun** विकसित और अनुरक्षित करते हैं।

- [आर्किटेक्चर](#architecture)
- [मॉड्यूल](#modules)
- [मुख्य प्रतीक](#symbols)
- [कॉल प्रवाह](#flows)
- [प्रवेश बिंदु](#entrypoints)
- [प्रभाव](#impact)
- [विस्तार](#extension)
- [MCP, निगरानी और साझा इंटरफ़ेस](#integrated-features)
- [स्रोत और सत्यापन](#validation)
- [रिपॉज़िटरी संरचना](#repository-organization)

- [लाइसेंस और तृतीय-पक्ष सूचनाएँ](#license)

<a id="architecture"></a>

## आर्किटेक्चर

एक सर्वर और एक वेब एप्लिकेशन। फ़ैक्टरी केवल `dgs_model_qwen35` बनाती है; Qwen3.8-27B उसी आर्किटेक्चर को साझा करता है। DeltaNet कर्नेल तथा साझा विज़न सहायक वाला Qwen प्रोजेक्टर बनाए रखे गए हैं। MoE, पुराने GGUF और अन्य प्रोफ़ाइल अस्वीकृत हैं। अनुमत सूची HTTP उपनामों से स्वतंत्र मूल मेटाडेटा पढ़ती है, फिर परतें, एम्बेडिंग, FFN और हेड जाँचती है। केवल आयामों से Qwen3.5-27B और Qwen3.8-27B में अंतर नहीं किया जा सकता। GGUF की घोषित पहचान का क्रिप्टोग्राफ़िक प्रमाणीकरण नहीं होता। GGML और बाहरी निर्भरताएँ अपने वर्तमान मूल ऑपरेशन और इंटरफ़ेस रखते हैं।

<a id="modules"></a>

## मॉड्यूल

| मॉड्यूल | पथ | उत्तरदायित्व | निर्भरताएँ | उपयोग करने वाले |
|---|---|---|---|---|
| नीति | `backend/src/dgs-model-policy.*` | Qwen की अनुमत सूची | GGUF, hparams | मॉडल लोडर |
| Qwen | `backend/src/models/` | सघन ग्राफ़ | DeltaNet, GGML | सर्वर |
| विज़न | `backend/tools/mtmd/` | Qwen चित्र/वीडियो | GGML, कोडेक | सर्वर |
| चैट | `backend/common/` | Jinja, XML टूल, सैंपलिंग | मॉडल, vendor | सर्वर |
| HTTP | `backend/tools/server/` | रूट, स्लॉट, टूल | common, mtmd, UI | निष्पादन फ़ाइल |
| वेब | `frontend/` | वर्तमान इंटरफ़ेस और संग्रहण | निश्चित संस्करण के npm पैकेज | ब्राउज़र |
| बिल्ड | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | कंपाइल और समाहित करना | कंपाइलर, npm, Python | परिनियोजन |
| शेल MCP | backend/tools/server/server-shell-mcp.* | MCP कमांड और पासवर्ड अनुरोध | HTTP, POSIX, common_json | सर्वर, MCP वेब |
| निगरानी | backend/tools/server/server-monitor.* | CPU/GPU टेलीमेट्री और मेमोरी समेकन | procfs, sysfs, वैकल्पिक NVML | server-context, राउटर, वेब |
| MTP वेट | backend/src/dgs-ext.h; dgs-model.cpp | सघन/MTP आवंटन का विभाजन | GGML बफ़र मेटाडेटा | निगरानी स्नैपशॉट |
| साझा सेटिंग्स | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | एटॉमिक सेटिंग्स दस्तावेज़ | फ़ाइल सिस्टम, API, SettingsService | सभी वेब स्टेट स्टोर |
| API दस्तावेज़ | backend/tools/server/server-api-doc.*; frontend/api.html | वर्तमान OpenAPI और Swagger | रूट रजिस्ट्री, स्कीमा, vendor/swagger-ui | /api/doc/ |
| भाषाएँ | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 कैटलॉग और उत्तर निर्देश | JSON, साझा सेटिंग्स | वेब घटक, चैट सेवा |
| कतार सूचनाएँ | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | अनुरोध पर कतार घटनाएँ | कार्य और उत्तर कतारें | स्ट्रीमिंग चैट इंटरफ़ेस |
| वेब संसाधन | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | इंटरफ़ेस बिल्ड और समाहित करना | npm, Python | `dgs-ui` |
| लाइसेंस वितरण | `backend/CMakeLists.txt`, `build/licenses/` | परियोजना और तीसरे पक्ष की शर्तें अलग रखना | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## मुख्य प्रतीक

| प्रतीक | फ़ाइल:पंक्ति | कार्य |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | GGUF पहचान सत्यापित करना |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | सघन आयाम सत्यापित करना |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | MoE टेंसर अस्वीकार करना |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | अनुमत मॉडल बनाना |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | Qwen आयाम लोड करना |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | Jinja रेंडर करना |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | Qwen विज़न ग्राफ़ बनाना |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | सर्वर चालू करना और रूट दर्ज करना |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | MCP JSON-RPC भेजना |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | शेल कॉल सत्यापित और निष्पादित करना |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | sudo पासवर्ड अनुरोध आगे भेजना |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | हार्डवेयर और मॉडल मेमोरी समेकित करना |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | सघन मेमोरी खंड प्रकाशित करना |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | जोड़े गए MTP वेट गिनना |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | साझा सेटिंग्स एटॉमिक रूप से सहेजना |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | OpenAPI और Swagger प्रदान करना |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | कतार स्थिति भेजना |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | क्षेत्रीय भाषा निर्देश जोड़ना |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | इस्तेमाल किए पासवर्ड स्ट्रिंग मिटाना |

<a id="flows"></a>

## कॉल प्रवाह

लोड: `fDgsModelCreate` → MoE अस्वीकार → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → टेंसर/ग्राफ़। चैट: HTTP → कतार/स्लॉट → Jinja → टोकनाइज़ेशन → इन्फ़रेंस → स्ट्रीमिंग उत्तर। वेब: बिना माइग्रेशन `DgsUi` स्कीमा का सीधा आरंभ। बिल्ड: npm → `frontend/dist` → `frontend/embed.py` → समाहित लाइब्रेरी। ड्राफ़्ट मॉडल भी यही नीति मानते हैं; MTP अनुमत मॉडल की भविष्यवाणी परतों का उपयोग करता है।

<a id="entrypoints"></a>

## प्रवेश बिंदु

| रूट | हैंडलर | फ़ाइल |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | वर्तमान पूर्णता रूट | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | वर्तमान API रूट | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | सर्वर/राउटर | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | समाहित वेब | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | स्नैपशॉट और राउटर समेकन | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | प्रीफ़िक्स और सहेजी गई भाषा | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## प्रभाव

नीति बदलने का असर हर लोड पर पड़ता है, जिसमें ड्राफ़्ट और केवल शब्दावली का लोड भी शामिल है। GGUF मेटाडेटा का नाम बदलने से पहचान विफल हो सकती है; उसकी जगह फ़ाइलनाम न दें। वर्तमान GGUF कुंजियों या साझा ऑपरेशन में बचे ऐतिहासिक नाम पुराने मॉडल सक्षम नहीं करते। हटाई API, माइग्रेशन या उपनाम वापस न लाएँ। स्कीमा बदलाव में फ़्रंटएंड और बैकएंड को साथ बदलना चाहिए।

<a id="extension"></a>

## विस्तार

नया मॉडल जोड़ने के लिए आर्किटेक्चर, मेटाडेटा और आयामों सहित अनुमत सूची पर स्पष्ट निर्णय चाहिए। केवल नाम देखकर मिलान न बढ़ाएँ और MoE वापस न जोड़ें। नए एंडपॉइंट सर्वर में दर्ज करें और वेब प्रकार बदलें। README, MANUAL और CODE को इंटरफ़ेस की सभी 15 भाषाओं में अद्यतन रखें।

<a id="validation"></a>

## स्रोत और सत्यापन

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

ऊपर के आधिकारिक कॉन्फ़िगरेशन प्रोफ़ाइल परिभाषित करते हैं। जाँच केवल कंपाइलेशन और स्थिर विश्लेषण से होती है; नियम परीक्षण रोकते हैं। मॉडल इन्फ़रेंस और थ्रूपुट मापन नहीं हुए। संरक्षित `.git` फ़ोल्डर स्वामी स्वयं हटाता है; एजेंट Git नहीं चलाता।



<a id="integrated-features"></a>

## MCP, निगरानी और साझा इंटरफ़ेस

शेल: CLI विकल्प → शेल की खोज → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → अनुमति जाँच → `tools/call` → स्वतंत्र प्रक्रिया समूह → JSON परिणाम और हर कॉल का JSONL → stdin/stdout/stderr प्रदर्शन। askpass के अलग चैनल में कॉल ID और निजी Unix socket हैं। वेब संवाद पासवर्ड मॉडल के संदेशों में नहीं डालता। `common_json::mWipeString` उपयोग के बाद सहायक की संवेदनशील JSON स्ट्रिंग पर लिखकर उन्हें मिटाता है।

निगरानी: मॉडल लोड → MTP वेट विभाजन → मुख्य लूप का स्नैपशॉट, इन्फ़रेंस के दौरान अधिकतम एक बार प्रति सेकंड → mutex-सुरक्षित प्रति → `/api/monitor*` → दिखाई देने वाला पृष्ठ हर दो सेकंड में पूछता है। मुख्य और अलग ड्राफ़्ट मॉडल के वेट एक ही बार गिने जाते हैं। विशेषज्ञों के बफ़र, कैश और प्रीफ़ेच श्रेणियाँ शामिल नहीं हैं। पोलिंग इन्फ़रेंस कार्य नहीं डालती और सोए मॉडल नहीं जगाती। हार्डवेयर उपयोग इन्फ़रेंस स्नैपशॉट से स्वतंत्र है।

सेटिंग्स: ब्राउज़र API कुंजी → `/api/ui/settings` → `UiSettingsStore.mLoad` → स्टोर आरंभ → अंतिम बदलाव के बाद छोटी देरी से पूरा दस्तावेज़ सहेजना। दस्तावेज़ में `config`, `userOverrides`, `language` और `preferences` हैं; प्राथमिकताएँ अपने स्टोर की सीरियलाइज़ेशन बनाए रखती हैं। साझा बदलाव रीलोड पर दिखते हैं, लाइव सहयोगी संपादन नहीं है; अंतिम पूरे दस्तावेज़ का सेव लागू होता है। बातचीत, टैब और स्ट्रीम पुनरारंभ कर्सर स्थानीय रहते हैं।

भाषा: `/api/config.js` → JSON कैटलॉग → इंटरफ़ेस। `fApplyResponseLanguage` केवल भेजे जाने वाले संदेशों की प्रतियाँ बदलता है; सामान्य चैट, टूल, शीर्षक और पूर्व-भरण इसे उपयोग करते हैं। जोड़े गए DGS पाठ en-US/en-GB/es-ES/es-AR में हैं; अन्य स्थानीय रूपों में आयातित कैटलॉग की प्रविष्टि न मिलने पर en-US लिया जाता है।

कतार: `dgs_client=webui` → पूर्णता स्कीमा → स्रोत के अनुसार काउंटर → `server_task_result_queue` → `dgs_queue` SSE → हर बातचीत का `QueueStore` → लंबित उत्तर लेबल। OpenAI Responses और Anthropic के घटना प्रारूपों को ये अतिरिक्त घटनाएँ नहीं भेजी जातीं। बिना चिह्न वाले API क्लाइंट अप्रभावित हैं।

API का डिफ़ॉल्ट आधार `/api` है, जिसे केवल `/api` के नीचे बदला जा सकता है; वेब `/` पर रहता है। हैंडलर सापेक्ष पथ लेते हैं, राउटर की सहायक प्रक्रियाएँ हमेशा `/api` उपयोग करती हैं और वेब प्रकाशित प्रीफ़िक्स जोड़ता है। API कॉल service worker कैश के बजाय नेटवर्क का उपयोग करती हैं। पथ बदलने पर राउटर अग्रेषण, क्लाइंट URL और OpenAPI साथ बदलने चाहिए।

विस्तार: शेल उम्मीदवार और उसका फ़्रंटएंड टूल नाम साथ जोड़ें; मेमोरी खंड को सघन स्नैपशॉट, API प्रकार और रंग समूह में एक साथ जोड़ें; UI पाठ JSON कैटलॉग में जोड़ें। सघन अनुमत सूची और वर्तमान प्रारूप बनाए रखें। बदलाव के बाद बैकएंड और फ़्रंटएंड कंपाइल करें; नियमों के अनुसार परीक्षण न लिखें और न चलाएँ।


एकीकृत सुविधाओं की जाँच: CPU Release, LTO और 70 समाहित वेब संसाधनों का बिल्ड सफल रहा। Svelte/TypeScript में शून्य त्रुटियाँ और चेतावनियाँ। वेब बंडल सलाहकारी आकार सीमा से बड़ा है; GCC/CMake ने पहले से मौजूद चेतावनियाँ और फ़ाइल सिस्टम घड़ी का अंतर भी बताया। परीक्षण, सर्वर संचालन, मॉडल इन्फ़रेंस या GPU बिल्ड नहीं हुए।

<a id="repository-organization"></a>

## रिपॉज़िटरी संरचना

```text
backend/
  CMakeLists.txt
  cmake/  common/  ggml/  include/  src/  vendor/
  tools/
    CMakeLists.txt
    mtmd/
    server/
build/
  CMakeLists.txt
  build-local.sh
  licenses/
deploy/
  scripts/ui-assets.cmake
doc/
  CODE*.md  MANUAL*.md  README.*.md
frontend/
  CMakeLists.txt  sources.cmake  embed.py
  src/  static/  scripts/
_/build/bin/dgs-server
```

बिल्ड प्रवाह: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, फ़्रंटएंड, सर्वर। UI: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → समाहित लाइब्रेरी। wrapper और backend दोनों CMake स्रोत-रूट हो सकते हैं। फ़ाइलें स्थानांतरित करने पर include, CMake पथ, निर्भरता सूची और दस्तावेज़ साथ बदलें।


संरचना की जाँच: build/ और backend/ से CMake कॉन्फ़िगरेशन सफल। build/ प्रवेश से CPU Release, LTO और समाहित UI बिल्ड सफल। Svelte/TypeScript में त्रुटि या चेतावनी नहीं; Bash सिंटैक्स और स्थानीय दस्तावेज़ लिंक जाँचे गए। परीक्षण या मॉडल इन्फ़रेंस नहीं हुए।


<a id="license"></a>

## लाइसेंस और तृतीय-पक्ष सूचनाएँ

NiPeGun के योगदान [PolyForm Noncommercial 1.0.0](../LICENSE) के अंतर्गत उपलब्ध हैं। शामिल तृतीय-पक्ष कोड अपने लाइसेंस और कॉपीराइट सूचनाएँ बनाए रखता है; ये [build/licenses/](../build/licenses/README.md) में एकत्रित हैं।

मुख्य LICENSE का पाठ अक्षरशः सुरक्षित है। विशेष रूप से ggml लेखकों का MIT नोटिस `build/licenses/LICENSE-llama.cpp` में रहता है; Swagger UI का Apache लाइसेंस और NOTICE भी सुरक्षित हैं। सूची में शामिल HTTP, JSON, हैश, मीडिया, subprocess, base64, CPU कर्नेल और गणितीय घटकों की सूचनाएँ भी हैं। निर्भरता स्रोत और वेब पैकेज में मौजूद नोटिस लागू रहते हैं। मॉडल वेट और सिस्टम लाइब्रेरी की अपनी शर्तें हैं।

डिफ़ॉल्ट डेटा संरचना में `cmake --install _/build --prefix PREFIX`, परियोजना का LICENSE और `build/licenses/` को `PREFIX/share/doc/dgs-server/licenses/` में कॉपी करता है। डिफ़ॉल्ट प्रीफ़िक्स `/usr/local` से स्थान `/usr/local/share/doc/dgs-server/licenses/` बनता है; उपयोगकर्ता स्थापना में `~/.local` लिया जा सकता है। `Licenses` घटक अलग से स्थापित हो सकता है। निष्पादन फ़ाइल स्वयं वितरित करते समय ये दस्तावेज़ साथ दें।

वितरण: परियोजना LICENSE और तृतीय-पक्ष सूचनाएँ → CMake घटक `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`। निर्भरता बदलने पर संबंधित सूचना की समीक्षा करें। उत्पाद का नाम और API प्रकार DGS उपयोग करते हैं; तृतीय-पक्ष श्रेय और GGUF पहचानकर्ता नहीं बदले जाते। अनुपयोगी बाहरी कैटलॉग अडैप्टर और असमर्थित मॉडल प्रीसेट हटाए गए; मॉडल खोज मौजूदा Hugging Face सेवा से होती है।

पहचान और लाइसेंस की जाँच: CPU Release, LTO और समाहित UI बिल्ड सफल; Svelte/TypeScript में शून्य त्रुटि और चेतावनी। उत्पन्न CMake स्थापना में मुख्य लाइसेंस और संकलित नोटिस शामिल हैं। मुख्य LICENSE संदर्भ के बाइट-दर-बाइट समान है। परीक्षण, इन्फ़रेंस या उत्पादन स्थापना नहीं हुई।
