<div dir="rtl">

# dense-gguf-server

מסמך README זה זמין גם בשפות הבאות:\
[de-DE](README.de-DE.md), [en-GB](README.en-GB.md), [en-US](../README.md), [es-AR](README.es-AR.md), [es-ES](README.es-ES.md), [fr-FR](README.fr-FR.md), **he-IL**, [hi-IN](README.hi-IN.md), [it-IT](README.it-IT.md), [ja-JP](README.ja-JP.md), [ko-KR](README.ko-KR.md), [pt-BR](README.pt-BR.md), [pt-PT](README.pt-PT.md), [ru-RU](README.ru-RU.md), [zh-CN](README.zh-CN.md)

## מהו הפרויקט

הריצו מודלי GGUF צפופים מקומית באמצעות צ'אט בדפדפן ו־API מסוג HTTP להסקה.

כמו llama.cpp, אבל נבנה בדיוק של אזמל ניפגוניאני כדי להשיג את מהירות ההסקה הגבוהה ביותר האפשרית.

## צילומי מסך

טרם נוספו צילומי מסך.

## פריסה

ב־**Debian**, לאחר בניית הפרויקט, הריצו את הפקודות הבאות בתור `root` מתיקיית השורש שלו:

<div dir="ltr">

```bash
cmake --install _/build --prefix /usr/local
dgs-server -m /path/to/Qwen3.5-4B.gguf --host 127.0.0.1 --port 8080 --local-shell-mcp-server
```

</div>

פתחו את `http://127.0.0.1:8080`. ה־API משתמש ב־`/api`, לקוחות התואמים ל־OpenAI משתמשים ב־`/api/v1`, ו־Swagger זמין ב־`/api/doc/`. משקלי המודלים מסופקים בנפרד; השתמשו ב־`--api-key` כשנותנים גישה ללקוחות אחרים.

## בנייה

תחילה התקינו את התלויות בתור `root` ב־Debian 13:

<div dir="ltr">

```bash
apt-get update
apt-get install -y build-essential cmake ninja-build pkg-config libssl-dev python3 nodejs npm
```

</div>

לאחר מכן הריצו מתיקיית הפרויקט עם חשבון המשתמש הרגיל:

<div dir="ltr">

```bash
bash build/build-local.sh
./_/build/bin/dgs-server -m /path/to/Qwen3.5-4B.gguf
```

</div>

הסקריפט בונה את המקורות המקומיים בתוך `_/build/`. האפשרות `DGS_BUILD_JOBS=8` קובעת את מספר המשימות המקביליות. ארגומנטים נוספים מועברים ל־CMake, למשל `-DGGML_CUDA=ON` או `-DGGML_VULKAN=ON` כאשר ה־SDK המתאים מותקן. [התלויות ואפשרויות הבנייה](MANUAL.he-IL.md#local-build) חייבות להיות זמינות מראש.

## קריאת התיעוד

עיינו במסמכים הבאים:

- [doc/MANUAL.he-IL.md](MANUAL.he-IL.md) לבנייה, התקנה ושימוש ב־DGS, הגדרת MCP וממשק הדפדפן, ואבחון תקלות.
- [doc/CODE.he-IL.md](CODE.he-IL.md) להבנת המנוע הצפוף, ארגון המקורות וחוזי הבנייה של צד השרת וצד הדפדפן.

## תמיכה בפרויקט

- אפשר **[להזמין ממני עבודה](mailto:nipegun@gmail.com?subject=About%20the%20dense-gguf-server%20repo)** להוספת יכולות חדשות.
- **GitHub Sponsors.** אפשר גם לתמוך בפרויקט כספית דרך GitHub: [github.com/sponsors/nipegun](https://github.com/sponsors/nipegun).

</div>
