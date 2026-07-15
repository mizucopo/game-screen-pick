# PROTOTYPE — reportと出力画像の公開契約

> 同じVideo Set選定結果を、利用者の確認とselector改善の両方に使える公開成果物として、どの情報階層で見せるべきか。

これはIssue #168の判断用throwaway prototypeであり、production実装ではない。架空の1 runを使い、構造が異なる3種類の`report.md`をブラウザで切り替える。

## Variants

- **A — Gallery first**: 選択画像と採用理由を最初に見せ、技術診断を末尾へ置く。
- **B — Timeline first**: source videoごとの進行と採用位置を最初に見せる。
- **C — Audit first**: 選定funnel、near miss、score内訳、stage provenanceを最初に見せる。

3案は同じ架空データと共通の[`report.sample.json`](report.sample.json)を使う。Markdown sampleも個別に保持する。

## Decision so far

- `report.md`は**A — Gallery first**を本文構造にする。
- **C — Audit first**のselection funnel、near miss、score内訳、Stage provenanceを末尾appendixへ統合する。
- **B — Timeline first**のsource videoとVideo Timeは各画像と`report.json`に保持するが、動画別の長大なtimelineは標準`report.md`に展開しない。
- Output Folderは`images/`、`report.json`、`report.md`で構成する。
- Frame Candidate IDはVideo Fingerprintと正確なVideo Timeからversion付きSHA-256 derivationで作る`frm_<64桁digest>`とし、完全値を`report.json`の安定identityにする。
- Selected Image Output Nameは`<最低4桁の全体選択順>_<scene_slug>_<Frame Candidate ID digest先頭12文字>.<ext>`とし、同一run内で短縮digestが衝突した場合だけ64文字まで延長する。
- Selected Image Encodingは元frameの解像度を維持した非可逆WebP quality 95に固定し、埋め込みmetadataを除去する。v1では形式やqualityを設定項目にしない。
- Report Source PathはVideo Input Folderからの相対pathを`/`区切りへ正規化し、`..`と絶対pathを許可しない。`report.md`と`report.json`ではVideo Order、Video Source IDと併記する。
- Report Video Timeは`report.md`で24時間を超えても折り返さない`HH:MM:SS.mmm`をhalf-up表示し、`report.json`ではsource/origin PTS、time base、既約分数のoffset secondsを正本にする。frame indexとfloat秒は載せない。
- Report Context EvidenceにはCue ID、source kind、正確なVideo Time範囲、reliability、Context Cue Relevanceだけを載せる。字幕・音声文字起こしの本文はcacheだけに保持し、公開成果物では引用しない。
- Report Video Fingerprintはalgorithm名と動画全体SHA-256の完全な64桁を`report.json`だけに載せる。`report.md`は短いVideo Source ID、Video Order、Report Source Pathだけを表示する。
- Report Near Miss Setは全未採用理由を集計し、各理由の代表を最低1件含めて残りを反実仮想utility順に選ぶ。`report.json`は`min(未採用総数, 100, max(20, 要求枚数×2))`件、`report.md`はその先頭最大10件までとする。
- Report Schema Versionは`game-screen-pick/report`のSemantic Versionとし、初版を`1.0.0`にする。breaking changeはmajor、additive changeはminor、構造を変えない修正はpatchとし、未知majorだけを拒否する。過去schemaと既存reportは書き換えず、Markdownはmachine-readable contractにしない。
- Atomic Output Publicationは同じ親filesystem上の隠しstaging Folderで全成果物を生成・検証・flushし、directory rename 1回で公開する。fatal errorではOutput Folderを公開せず、Selection Shortfallはwarning付きで公開する。非atomic fallbackは設けない。
- Report Selection ExplanationはCandidate Annotationの画像要約・Representative Frame理由と、reason code・score内訳からlocal生成するselector採用理由を分離する。JSONでは`annotation`と`selection`を別objectにし、単一のmodel生成reason、confidence、内部推論を載せない。
- Report Spoiler Disclosureはrisk levelを常時表示し、`none`以外のmodel由来evidence summaryだけをMarkdownの閉じた`details`に置く。画面内文章とContext Cueを引用せず、selectorのSpoiler Penaltyとは別fieldにする。
- Report Stage ProvenanceはJSONで各Stageの完全なfingerprint・上流fingerprint、cache・再計算・duration・attempt・validation・token、正規化設定、tool・runtime・model digest、contract versionを保持する。Markdownは短縮fingerprintと主要件数・時間だけを表示し、path、環境変数、credential、prompt本文、raw response、stack traceを載せない。
- Canonical Selection Reportはschema検証済みobjectを`report.json`へserializeする唯一のmachine-readable正本とする。`report.md`は同じobjectだけから決定的に生成し、cacheやmodelを再参照しない。画像ID・path・件数・理由の不一致やMarkdown生成失敗はfatalとする。
- Report Image Embedは実際の選択WebPをrelative pathでinline表示し、同じ画像へのlinkで原寸を開ける。thumbnailは別生成せず、alt textは選択順とScene Display Nameだけにする。

## Run

```bash
uv run python -m http.server 8765 --directory prototypes/issue_168_report_contract
```

- A: <http://127.0.0.1:8765/?variant=A>
- B: <http://127.0.0.1:8765/?variant=B>
- C: <http://127.0.0.1:8765/?variant=C>

画面下部のswitcherまたは左右矢印keyで切り替えられる。

contract sampleの整合性は次で検証する。

```bash
uv run python prototypes/issue_168_report_contract/verify_contract.py
```

## Prototype artifacts

- [`variant-a-gallery.md`](variant-a-gallery.md)
- [`variant-b-timeline.md`](variant-b-timeline.md)
- [`variant-c-audit.md`](variant-c-audit.md)
- [`report.sample.json`](report.sample.json)
- [`verify_contract.py`](verify_contract.py)

表示画像はCSSで作ったplaceholderであり、実際のゲーム画像や個人pathは含まない。JSON sampleのfield配置は最終schemaへ統合する前のprototypeである。
