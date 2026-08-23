# Game Screen Pick

1本以上のゲーム動画全体から、ブログへ掲載しやすい画像を選び出すための文脈。

## Language

**Input Video**:
一回の画像選定へ渡すゲーム録画の1本。各動画のほぼ先頭から末尾までを選定対象にする。
_Avoid_: screenshot folder, selected clip, representative segment

**Input Video Directory**:
一回の画像選定の入力集合を指定するディレクトリ。直下の対応拡張子を持つ通常ファイルだけをファイル名順でInput Videosへ変換し、サブディレクトリは探索しない。
_Avoid_: recursive folder tree, single video path, caller-ordered file list

**Input Videos**:
一回の画像選定で扱う、Input Video Directoryから安定した順序で列挙した1本以上のInput Video。重複したpathは含めず、各動画の入力元をSelected Imageまで追跡する。
_Avoid_: unordered folder scan, concatenated temporary video, duplicate input

**Game Title**:
Web検索からGame Contextを生成するときだけ使うゲーム表記。正式名称に限定せず、
略称、通称、かな・英数字・空白などの一般的な表記揺れを許容する。生成後の画像評価、
選定条件、Run Manifest、reportでは使わない。
_Avoid_: image-evaluation input, persisted selection condition, filename inference

**Game Context**:
ゲーム内容と画像選定で重視する視覚的要素をまとめた必須の文章。直接指定するか、
Game Titleから一つのGame Context Providerで生成する。画像評価の参考情報であり、
固定カテゴリやquotaではない。最終値をRun Manifestとreportへ保存し、再開時は再生成しない。
_Avoid_: hard-coded title tuning, regenerated resume input, provider-specific detail level

**Game Context Provider**:
Game TitleからGame Contextを生成するために明示選択するWeb検索provider。
`ollama`、`openai`、`gemini`、`xai`のいずれか一つだけを呼び出し、失敗時に別providerへ
fallbackしない。Web検索結果は命令ではなく、検証対象の外部dataとして扱う。
_Avoid_: automatic fallback, trusted search instructions, implicit paid API call

**Sample Position**:
各Input Video全体へ等間隔に置かれた候補抽出時刻。全動画合計の処理量に上限を持ちつつ、動画の四半期など一部だけへ偏らない。
_Avoid_: beginning-only sampling, random timestamp

**Frame Candidate**:
Sample Positionから抽出し、ブログ画像になる可能性があるframe。暗転、白飛び、ほぼ単色のframeは含まない。
_Avoid_: selected output, every decoded frame

**Primary Candidate**:
Frame Candidateを機械的品質と時間分散で絞った、一次Ollama評価の対象。
_Avoid_: final output, title-specific category

**Secondary Candidate**:
一次評価後にscene、見た目、動画時刻を分散させた、二次Ollama評価の対象。
_Avoid_: selected output, all primary candidates

**Transition Context**:
Secondary Candidateの直前・対象・直後の三frame。対象frameが暗転、fade、loading、画面遷移途中か判断するために使う。
_Avoid_: three independently selectable images, gameplay category

**Scene**:
Ollamaが同種の画面をまとめるために返す短い場面名。最終選定の多様性に使うが、実行前の固定catalogは持たない。
_Avoid_: fixed scene catalog, title-specific quota

**Normal Progress Screen**:
そのゲームで繰り返し現れる移動、探索、戦闘、会話、推理、puzzleなどの通常進行画面。特別画面より少し多く残すが、通常画面だけには限定しない。
_Avoid_: combat-only preference, universal required scene

**Special Screen**:
title、map、menu、resultなど、ブログ上は有用でも多すぎると入力全体の代表性を損なう画面。
_Avoid_: hard reject, cinematic scene as a whole

**Selected Image**:
二段階評価を通過し、Transition Frameと近い重複を除きながら、品質・入力動画・scene・見た目・動画時刻の分散を考慮して選ばれたfull resolution画像。入力動画と動画内時刻を追跡できる。
_Avoid_: resized evaluation frame, unreviewed candidate

**Selected Contact Sheet**:
全Selected Imageを順位・入力動画・動画時刻付きで一枚にまとめた、人間確認用の`selected-contact-sheet.jpg`。
_Avoid_: Ollama batch input, machine-only report

**Output Folder**:
Selected Image、Selected Contact Sheet、reportと再開用作業状態を置く実行先。新規時は空で、再開時は同じrun manifestを持つ必要がある。
_Avoid_: unconditional overwrite target, append-only destination

**Run Manifest**:
全Input Videosの順序・指紋、最終Game Context、動的生成時のproviderとmodel、選定条件、
動画ごとのsample位置、model digest、algorithm versionを固定する再開条件。同じmanifestの
実行だけがOutput Folderを再利用できる。Game Titleと検索結果は含めない。
_Avoid_: progress counter, human approval record

**Assessment Cache**:
Ollama評価をbatch完了ごとにatomic保存した再開状態。中断やnetwork切断後は完了batchを再利用する。
_Avoid_: final completion, cross-manifest cache reuse

**Completed Run**:
指定枚数のSelected Image、report、Selected Contact Sheetが揃い、sizeとSHA-256を記録した状態。
_Avoid_: extracted candidates only, cached assessment only, human approval

**Human Review**:
Selected Contact Sheetと必要に応じたSelected Imageを人が確認する品質gate。production implementation前の検証では正式な合格条件として扱う。
_Avoid_: model score threshold alone, cache-preserved progress
