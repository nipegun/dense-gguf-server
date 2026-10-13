<div dir="rtl">

# DGS — מפת הקוד

[de-DE](CODE.de-DE.md) · [en-GB](CODE.en-GB.md) · [en-US](CODE.md) · [es-AR](CODE.es-AR.md) · [es-ES](CODE.es-ES.md) · [fr-FR](CODE.fr-FR.md) · **he-IL** · [hi-IN](CODE.hi-IN.md) · [it-IT](CODE.it-IT.md) · [ja-JP](CODE.ja-JP.md) · [ko-KR](CODE.ko-KR.md) · [pt-BR](CODE.pt-BR.md) · [pt-PT](CODE.pt-PT.md) · [ru-RU](CODE.ru-RU.md) · [zh-CN](CODE.zh-CN.md)

Dense GGUF Server הוא פרויקט עצמאי שמפותח ומתוחזק בידי **NiPeGun**.

- [ארכיטקטורה](#architecture)
- [מודולים](#modules)
- [סמלים](#symbols)
- [זרימות קריאה](#flows)
- [נקודות כניסה](#entrypoints)
- [השפעה](#impact)
- [הרחבה](#extension)
- [MCP, ניטור וממשק משותף](#integrated-features)
- [מקורות ואימות](#validation)
- [מבנה המאגר](#repository-organization)

- [רישיון והודעות צד שלישי](#license)

<a id="architecture"></a>

## ארכיטקטורה

שרת אחד ויישום דפדפן אחד. המפעל יוצר רק את `dgs_model_qwen35`; גם Qwen3.8-27B משתמש בארכיטקטורה זו. גרעיני DeltaNet ומקרן Qwen עם עזר הראייה המשותף נשמרים. MoE, גרסאות GGUF ישנות ופרופילים אחרים נדחים. רשימת ההרשאה קוראת מטא־נתונים מקוריים ללא תלות בכינויי HTTP ואז בודקת שכבות, embeddings,‏ FFN וראשי קשב. ממדים לבדם אינם מבדילים בין Qwen3.5-27B ל־Qwen3.8-27B. הזהות מוצהרת ב־GGUF ולא מאומתת קריפטוגרפית. GGML והתלויות החיצוניות שומרות על פעולות היסוד ועל הממשקים העדכניים שלהן.

<a id="modules"></a>

## מודולים

| מודול | נתיב | אחריות | תלוי ב־ | משמש את |
|---|---|---|---|---|
| מדיניות | `backend/src/dgs-model-policy.*` | רשימת Qwen מורשים | GGUF, hparams | טוען המודלים |
| Qwen | `backend/src/models/` | גרף צפוף | DeltaNet, GGML | שרת |
| ראייה | `backend/tools/mtmd/` | תמונות/וידאו Qwen | GGML, מקודדים | שרת |
| צ'אט | `backend/common/` | Jinja, כלי XML, דגימה | מודל, vendor | שרת |
| HTTP | `backend/tools/server/` | נתיבים, חריצים, כלים | common, mtmd, UI | קובץ הרצה |
| ווב | `frontend/` | ממשק ואחסון עדכניים | חבילות npm בגרסאות קבועות | דפדפן |
| בנייה | `build/CMakeLists.txt`, `backend/CMakeLists.txt`, `deploy/scripts/` | הידור והטמעה | מהדר, npm, Python | פריסה |
| MCP למעטפות | backend/tools/server/server-shell-mcp.* | פקודות MCP ובקשות סיסמה | HTTP, POSIX, common_json | שרת, ממשק MCP |
| ניטור | backend/tools/server/server-monitor.* | טלמטריית CPU/GPU וסיכום זיכרון | procfs, sysfs, NVML אופציונלי | server-context, נתב, ווב |
| משקלי MTP | backend/src/dgs-ext.h; dgs-model.cpp | חלוקת הקצאות צפופות/MTP | מטא־נתוני מאגרי GGML | תמונות מצב לניטור |
| הגדרות משותפות | backend/tools/server/server-ui-settings.*; frontend/src/lib/stores/ui-settings.svelte.ts | מסמך הגדרות אטומי | מערכת קבצים, API, SettingsService | כל מאגרי מצב הממשק |
| תיעוד API | backend/tools/server/server-api-doc.*; frontend/api.html | OpenAPI ו־Swagger עדכניים | רישום נתיבים, סכימה, vendor/swagger-ui | /api/doc/ |
| שפות | frontend/src/lib/i18n.ts; frontend/src/lib/locales/ | 15 קטלוגים והוראת תשובה | JSON, הגדרות משותפות | רכיבי ווב, שירות צ'אט |
| הודעות תור | backend/tools/server/server-{context,queue,task,schema}.*; frontend/src/lib/stores/queue.svelte.ts | אירועי תור לפי בקשה | תורי משימות ותשובות | ממשק צ'אט זורם |
| משאבי ווב | `deploy/scripts/ui-assets.cmake`, `frontend/sources.cmake`, `frontend/embed.py` | בניית הממשק והטמעתו | npm, Python | `dgs-ui` |
| הפצת רישיונות | `backend/CMakeLists.txt`, `build/licenses/` | הפרדה בין תנאי הפרויקט לתנאי צד שלישי | GNUInstallDirs | CMake install |

<a id="symbols"></a>

## סמלים

| סמל | קובץ:שורה | תפקיד |
|---|---|---|
| `fDgsRequireModelProfile` | `backend/src/dgs-model-policy.cpp:54` | אימות זהות GGUF |
| `fDgsValidateModelProfile` | `backend/src/dgs-model-policy.cpp:79` | אימות ממדי מודל צפוף |
| `fRequireDenseModel` | `backend/src/dgs-model.cpp:64` | דחיית טנזורי MoE |
| `fDgsModelCreate` | `backend/src/dgs-model.cpp:50` | יצירת מודל מותר |
| `mLoadArchHparams` | `backend/src/models/qwen35.cpp:5` | טעינת ממדי Qwen |
| `fCommonChatTemplatesApply` | `backend/common/chat.cpp:351` | עיבוד Jinja |
| `fClipGetGraphBuilder` | `backend/tools/mtmd/clip.cpp:922` | יצירת גרף הראייה של Qwen |
| `fDgsServer` | `backend/tools/server/server.cpp:42` | הפעלת השרת ורישום הנתיבים |
| `server_shell_mcp::mHandlePost` | `backend/tools/server/server-shell-mcp.cpp:1511` | ניתוב JSON-RPC של MCP |
| `server_shell_mcp::mToolsCallResult` | `backend/tools/server/server-shell-mcp.cpp:1343` | בדיקת קריאת מעטפת והרצתה |
| `fServerShellMcpAskpassMain` | `backend/tools/server/server-shell-mcp.cpp:1113` | העברת בקשות סיסמה של sudo |
| `fMonitorReport` | `backend/tools/server/server-monitor.cpp:725` | סיכום חומרה וזיכרון מודלים |
| `mBuildMonitorSnapshot` | `backend/tools/server/server-context.cpp:1041` | פרסום מקטעי זיכרון צפוף |
| `fDgsModelGetMtpBreakdown` | `backend/src/dgs-model.cpp:3323` | ספירת משקלי MTP מצורפים |
| `fUiSettingsMakePostHandler` | `backend/tools/server/server-ui-settings.cpp:185` | שמירה אטומית של הגדרות משותפות |
| `fRegisterApiDocumentation` | `backend/tools/server/server-api-doc.cpp:532` | הגשת OpenAPI ו־Swagger |
| `mSendQueueStatus` | `backend/tools/server/server-context.cpp:2897` | שליחת המיקום בתור |
| `fApplyResponseLanguage` | `frontend/src/lib/utils/response-language.ts:6` | הוספת הוראת שפה אזורית |
| `common_json::mWipeString` | `backend/common/json.cpp:36` | מחיקת מחרוזות סיסמה שנצרכו |

<a id="flows"></a>

## זרימות קריאה

טעינה: `fDgsModelCreate` → דחיית MoE → `fDgsRequireModelProfile` → Qwen → `fDgsValidateModelProfile` → טנזורים/גרף. צ'אט: HTTP → תור/חריץ → Jinja → חלוקה לטוקנים → הסקה → תשובה בזרימה. ווב: אתחול ישיר של סכימת `DgsUi` ללא המרות. בנייה: npm → `frontend/dist` → `frontend/embed.py` → ספרייה מוטמעת. מודלי טיוטה כפופים לאותה מדיניות; MTP משתמש בשכבות החיזוי של המודל המורשה.

<a id="entrypoints"></a>

## נקודות כניסה

| נתיב | מטפל | קובץ |
|---|---|---|
| `bash build/build-local.sh` | `fMain` | `build/build-local.sh` |
| `dgs-server -m` | `fDgsServer` | `backend/tools/server/server.cpp` |
| `/api/v1/chat/completions`, `/api/v1/completions` | נתיבי השלמה עדכניים | `backend/tools/server/server.cpp` |
| `/api/v1/embeddings`, `/api/v1/models` | נתיבי API עדכניים | `backend/tools/server/server.cpp` |
| `/api/health`, `/api/props`, `/api/slots`, `/api/models/*` | שרת/נתב | `backend/tools/server/server.cpp`, `server-models.cpp` |
| `/` | ממשק מוטמע | `backend/tools/server/server-http.cpp` |
| `/api/mcp/shell` | `server_shell_mcp::mHandlePost` | `server-shell-mcp.cpp` |
| `/api/mcp/shell/askpass` | `mHandleAskpassGet/Post` | `server-shell-mcp.cpp` |
| `/api/monitor`, `/api/monitor/memory`, `/api/monitor/usage` | תמונות מצב וסיכום הנתב | `server-context.cpp`, `server-models.cpp`, `server-monitor.cpp` |
| `/api/ui/settings` | `fUiSettingsMakeGetHandler/PostHandler` | `server-ui-settings.cpp` |
| `/api/config.js` | קידומת ושפה שמורה | `server-http.cpp` |
| `/api/doc/`, `/api/doc/openapi.json` | `fRegisterApiDocumentation` | `server-api-doc.cpp` |
| `/#/monitor` | `MonitorScreen` | `frontend/src/routes/monitor/+page.svelte` |

<a id="impact"></a>

## השפעה

שינוי המדיניות משפיע על כל טעינה, כולל טיוטות וטעינת אוצר מילים בלבד. שינוי שמות מטא־נתוני GGUF עלול למנוע זיהוי; אין להחליפם בשמות קבצים. שמות היסטוריים שנשארים כמפתחות GGUF או כפעולות יסוד משותפות אינם מפעילים תמיכה במודלים ישנים. אין להחזיר API, המרות או כינויים שהוסרו. שינויי סכימה דורשים עדכון מתואם בשני הצדדים.

<a id="extension"></a>

## הרחבה

הוספת מודל דורשת החלטה מפורשת על רשימת ההרשאה, הארכיטקטורה, המטא־נתונים והממדים. אין להרחיב זיהוי לפי שם בלבד או להחזיר MoE. רשמו נקודות קצה חדשות בשרת ועדכנו את טיפוסי הממשק. תחזקו README, MANUAL ו־CODE בכל 15 שפות הממשק.

<a id="validation"></a>

## מקורות ואימות

- [Qwen3.8-27B — config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)
- [Qwen3.5-9B — config.json](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
- [Qwen3.5-4B — config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-2B — config.json](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

קובצי התצורה הרשמיים לעיל מגדירים את הפרופילים. האימות משתמש רק בהידור ובניתוח סטטי; כללי הפרויקט אוסרים בדיקות. לא בוצעו הסקה או מדידות תפוקה. הבעלים מוחק ידנית את התיקייה המוגנת `.git`; הסוכן אינו משתמש ב־Git.



<a id="integrated-features"></a>

## MCP, ניטור וממשק משותף

מעטפת: אפשרות CLI → זיהוי מעטפות → `/api/props.shell_mcp_server` → `MCPStore.mSyncShellServer` → בדיקת הרשאה → `tools/call` → קבוצת תהליכים נפרדת → תוצאת JSON ו־JSONL לכל קריאה → תצוגת stdin/stdout/stderr. ערוץ askpass הצדדי משתמש במזהה קריאה וב־Unix socket פרטי; הממשק לעולם אינו מוסיף סיסמאות להודעות המודל. `common_json::mWipeString` דורס מחרוזות JSON רגישות של תהליך העזר לאחר שימוש.

ניטור: טעינת מודל → חלוקת משקלי MTP → תמונת מצב בלולאה הראשית, לכל היותר פעם בשנייה בזמן הסקה → עותק מוגן mutex → `/api/monitor*` → תשאול כל שתי שניות כשהדף גלוי. משקלי המודל הראשי והטיוטה הנפרדת נספרים פעם אחת. אין במוניטור מאגרי מומחים, מטמוני מומחים או קטגוריות קדם־שליפה. התשאול אינו יוצר משימות הסקה ואינו מעיר מודלים רדומים. שימוש החומרה נמדד בנפרד מתמונת מצב ההסקה.

הגדרות: מפתח API בדפדפן → `/api/ui/settings` → `UiSettingsStore.mLoad` → אתחול מאגרי מצב → שמירת המסמך כולו לאחר השהיה קצרה מהשינוי האחרון. במסמך יש `config`, `userOverrides`, `language` ו־`preferences`; ההעדפות שומרות על ייצוג הסריאליזציה של המאגר שלהן. השיתוף בין הדפדפנים מתעדכן בטעינה מחדש, ללא עריכה שיתופית חיה; השמירה האחרונה של כל המסמך קובעת. שיחות, לשוניות וסמני חידוש זרימה נשארים מקומיים.

שפה: `/api/config.js` → בחירת קטלוג JSON → ממשק. `fApplyResponseLanguage` משנה רק עותקים של הודעות יוצאות ומשמש צ'אט, כלים, כותרות ומילוי מוקדם. מחרוזות DGS החדשות כוללות en-US/en-GB/es-ES/es-AR; בשפות אחרות מוצג en-US אם חסרה רשומה בקטלוג שיובא.

תור: `dgs_client=webui` → סכימת השלמה → מונים לפי מקור → `server_task_result_queue` → אירועי SSE מסוג `dgs_queue` → `QueueStore` לכל שיחה → תווית תשובה ממתינה. פורמטי האירועים של OpenAI Responses ושל Anthropic אינם מקבלים את האירועים הנוספים. לקוחות API שלא סומנו אינם מושפעים.

בסיס ה־API הוא `/api` כברירת מחדל, וניתן לשינוי רק מתחת ל־`/api`; הממשק נשאר ב־`/`. המטפלים מקבלים נתיבים יחסיים, תהליכי הנתב משתמשים תמיד ב־`/api`, והממשק פותר את הקידומת שפורסמה. קריאות API יוצאות לרשת במקום למטמון service worker. שינויי נתיבים מחייבים עדכון משותף של העברת הנתב, יצירת כתובות הלקוח ו־OpenAPI.

הרחבה: הוסיפו יחד מעטפת מועמדת ואת שם הכלי שלה בממשק; הוסיפו מקטעי זיכרון יחד בתמונת המצב הצפופה, בטיפוסי ה־API ובקבוצות הצבע; הוסיפו טקסט לקטלוגי JSON. שמרו על רשימת המודלים הצפופים והפורמטים העדכניים. הידרו את שני הצדדים אחרי שינוי; אין להוסיף או להריץ בדיקות לפי הכללים.


אימות היכולות המשולבות: בניית CPU Release עם LTO ו־70 משאבי ווב מוטמעים הושלמה. Svelte/TypeScript דיווח על אפס שגיאות ואזהרות. חבילת הממשק חורגת מגודל ההמלצה; GCC/CMake דיווחו גם על אזהרות קיימות ועל הפרשי שעון במערכת הקבצים. לא בוצעו בדיקות, הפעלת השרת, הסקה או בניית GPU.

<a id="repository-organization"></a>

## מבנה המאגר

<div dir="ltr">

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

</div>

בנייה: `build/CMakeLists.txt` → `backend/CMakeLists.txt` → `backend/tools/CMakeLists.txt` → mtmd, צד דפדפן, שרת. ממשק: `frontend/CMakeLists.txt` → `frontend/sources.cmake` + `deploy/scripts/ui-assets.cmake` → npm/dist → `frontend/embed.py` → ספרייה מוטמעת. גם העטיפה וגם backend יכולים להיות שורשי המקור של CMake. העברת קבצים מחייבת עדכון של include, נתיבי CMake, רשימות תלויות ותיעוד יחד.


אימות המבנה: הגדרת CMake הצליחה מ־build/ ומ־backend/. בניית CPU Release עם LTO והממשק המוטמע הצליחה דרך build/. Svelte/TypeScript ללא שגיאות ואזהרות; תחביר Bash וקישורי התיעוד המקומיים נבדקו. לא בוצעו בדיקות או הסקת מודלים.


<a id="license"></a>

## רישיון והודעות צד שלישי

התרומות של NiPeGun מופצות לפי [PolyForm Noncommercial 1.0.0](../LICENSE). הקוד של צדדים שלישיים שומר על הרישיונות ועל הודעות זכויות היוצרים שלו, המרוכזים ב־[build/licenses/](../build/licenses/README.md).

קובץ LICENSE הראשי נשמר כלשונו. בפרט, הודעת MIT של מחברי ggml נשמרת ב־`build/licenses/LICENSE-llama.cpp`; Swagger UI שומר על רישיון Apache ועל NOTICE. הרשימה כוללת גם רכיבי HTTP,‏ JSON, גיבוב, מדיה, subprocess,‏ base64, גרעיני CPU וספריות מתמטיקה המשולבים בפרויקט. הודעות במקורות התלויות ובחבילות הווב נשארות בתוקף. למשקלי המודלים ולספריות המערכת יש תנאים משלהם.

במבנה הנתונים הרגיל, `cmake --install _/build --prefix PREFIX` מעתיק את LICENSE של הפרויקט ואת `build/licenses/` אל `PREFIX/share/doc/dgs-server/licenses/`. קידומת ברירת המחדל `/usr/local` מתקינה אותם ב־`/usr/local/share/doc/dgs-server/licenses/`; התקנה למשתמש יכולה לבחור `~/.local`. אפשר להתקין בנפרד את הרכיב `Licenses`. צרפו מסמכים אלה גם בעת הפצה ידנית של קובץ ההרצה.

הפצה: LICENSE של הפרויקט והודעות צד שלישי → רכיב CMake `Licenses` → `${prefix}/share/doc/dgs-server/licenses/`. שינוי תלות דורש בדיקה של ההודעה שלה. שם המוצר וטיפוסי ה־API משתמשים ב־DGS; שמות הייחוס של צדדים שלישיים ומזהי GGUF אינם משתנים. מתאם הקטלוג החיצוני שלא היה בשימוש והגדרות מודלים לא נתמכים הוסרו; גילוי מודלים ממשיך דרך שירות Hugging Face הקיים.

אימות זהות ורישיונות: בניית CPU Release עם LTO וממשק מוטמע הושלמה, ו־Svelte/TypeScript דיווח על אפס שגיאות ואזהרות. התקנת CMake שנוצרה כוללת את הרישיון הראשי ואת ההודעות שנאספו. LICENSE בשורש זהה להפניה ברמת הבייט. לא בוצעו בדיקות, הסקה או התקנה בייצור.

</div>
