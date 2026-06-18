# Game Screen Pick

ゲームスクリーンショットから、ブログで使いやすい画像を選び出すための文脈。

## Language

**Scene**:
ブログ用の画像選択で使う、画像内容を表すカテゴリ。ゲームジャンルや入力画像群に応じて決まる。
_Avoid_: play/event density bucket, fixed category

**Scene Slug**:
scene を表す小文字英数字の安定名。出力ファイル名、レポート、カテゴリ集計に使われる。
_Avoid_: localized category name

**Scene-numbered Output Name**:
選択された画像に付ける標準の出力ファイル名。scene slug と scene 内の連番で構成される。
_Avoid_: original filename output, optional rename mode

**Output Folder**:
選択された画像とレポートを書き出す実行ごとの保存先。処理開始前に空である必要がある。
_Avoid_: append destination, overwrite target, resumable output

**Scene Display Name**:
scene を人が読みやすいように表す日本語名。ブログ用の画像選択やレポート表示で使われる。
_Avoid_: filename prefix, report key

**Scene Catalog**:
入力画像群から見つけた、その実行で使う scene と scene selection role の一覧。3から8個の scene で構成され、分類の逃げ先として other を含む。各画像は scene catalog のいずれかの scene に分類される。
_Avoid_: fixed scene list, free-form per-image labels

**Scene Description**:
画像がその scene に分類された理由を、ブログ用の画像選択に役立つように短く説明する文章。
_Avoid_: internal reasoning, model trace

**Scene Selection Role**:
scene ごとに、最終選択での扱いを表す役割。値は `ordinary`、`cinematic`、`recurring_gameplay` の3種類で、other scene、その他や不明なroleは通常配分で扱う。
_Avoid_: scene label, manual quota, content reject reason, failure mode

**Scene Hint**:
scene catalog を作るときに、ユーザーがゲームジャンルやブログ画像選択の意図を補足する短い説明。
_Avoid_: fixed scene list, selection rule

**Selection Intent**:
ブログ画像として何を重視して選ぶかを表す実行ごとの意図。scene hint は selection intent を補足する入力であり、変わると scene catalog や画像分類も変わり得る。
_Avoid_: image analysis setting, cache option

**Quality Score**:
blog candidate がブログ画像としてどれだけ使いやすいかを表す評価値。scene の種類やゲームジャンルの指示ではなく、画像そのものの見やすさを表す。
_Avoid_: scene hint, user-facing mode, selection profile

**Blog Candidate**:
ブログ画像として選択する余地があるスクリーンショット。明らかな暗転、白飛び、単色画面、遷移フレームは含まない。
_Avoid_: all input images

**Selection Shortlist**:
blog candidate のうち、ブログに採用される可能性が高く、最終選別のためにscene分類へ進める画像群。
_Avoid_: all blog candidates, selected output

**Neutral Image Analysis**:
scene や selection intent に依存せず、画像そのものから得られる特徴と品質評価。画像の内容分類ではなく、blog candidate 判定や類似度判定の土台になる。
_Avoid_: scene classification, selection intent

**Transition Frame**:
シーン移動や画面切り替えの途中に現れる、ブログ画像として説明価値が低い一時的な画面。
_Avoid_: event scene, cutscene

**Cinematic Scene**:
ゲームの進行操作より演出、会話、イベントの見せ場を主に写した scene。ブログ画像として少量は有用だが、入力全体の代表性を崩さないよう通常 gameplay より控えめに扱う。
_Avoid_: transition frame, hard reject, movie frame

**Cinematic Soft Cap**:
すべての cinematic scene の合計選択枚数を通常は少量に抑えつつ、他の有用候補が足りない場合だけ超過を許す上限。重要な見せ場を残しながら、入力全体が cinematic に偏ることを防ぐ。
_Avoid_: per-scene cinematic quota, hard reject, fixed exclusion, exact quota

**Recurring Gameplay Pattern**:
戦闘UI、探索画面、パズル盤面など、ゲーム中に頻繁に表示される通常playの画面構造。同じ構図でも状態や進行の違いがブログ上の説明価値になるため、複数のvariantを選ぶ余地がある。
_Avoid_: duplicate image, cinematic scene, static menu

**Variant Expansion**:
recurring gameplay pattern で、同じ variant group から複数の画像を選ぶこと。要求選択枚数が多いほど強まり、同じ画面構造の中にある状態差や進行差を拾うために使う。
_Avoid_: duplicate flooding, one-representative-only selection, manual expansion mode

**Variant Group**:
同じ scene の中で、見た目や構図が近くブログ上の役割が重複する画像のまとまり。最終選択では原則として各 variant group から代表画像を1枚だけ選ぶが、recurring gameplay pattern では variant expansion の対象になる。
_Avoid_: scene, duplicate file

**Ollama Classification Failure**:
blog candidate を scene catalog の scene に分類できなかった状態。other に分類された画像とは区別され、最終選択の対象にはならない。
_Avoid_: other scene, rejected by content filter

**Ollama Catalog Fallback**:
scene catalog を作成できないときに、処理継続のため全 blog candidate を fallback scene に割り当てる代替状態。Ollama Classification Failure とは区別される。
_Avoid_: per-image classification failure, other scene

**Resumable Run**:
中断された画像選択を、同じ入力画像群と同じ選択意図で後から続ける実行。すでに得られた画像解析やscene分類の結果を再利用し、未処理の画像だけを進める。
_Avoid_: fresh run, output overwrite
