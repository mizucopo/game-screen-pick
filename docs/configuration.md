# 動画入力設定

> [!IMPORTANT]
> v1 schema、Effective Configuration resolver、model lifecycleは内部実装済みです。installed public CLIはIssue #190までscreenshot入力のままであり、現在の`--config`にはこのschemaを使用できません。

内部・test用adapterは、設定解決が成功した後だけ`run(EffectiveConfiguration) -> RunOutcome`境界を呼び出します。これにより、public cutover前もstrict schemaと優先順位を副作用なしで検証できます。

## 読み込みと優先順位

TOMLは`--config PATH`を指定した場合だけ読み込みます。current directory、home directory、Video Input Folderからの自動探索は行いません。

Effective Configurationは設定項目ごとに次の順で解決します。

1. 明示されたCLI値
2. 明示されたTOML値
3. 環境変数
4. 組み込み既定値

公開する環境変数は`OLLAMA_HOST`だけです。したがって`[ollama].host`をTOMLへ明示すると、`OLLAMA_HOST`よりTOMLが優先されます。CLI booleanは未指定・true・falseを区別し、`--no-recursive`でTOMLの`recursive = true`を上書きできます。

未知のsection、key、型、enum、範囲外の値、未対応の`config_version`はexit 2のusage/config errorです。既知keyのtypoを無視しません。config error時は入力探索、network access、cache reset、output作成を行いません。

## v1 schemaと既定値

完全な例は[`examples/video-selection.toml`](examples/video-selection.toml)です。省略したkeyには次の組み込み既定値を使います。

| Key | Type / constraint | Default |
|---|---|---|
| `config_version` | exact string `1.0.0`、必須 | なし |
| `input.recursive` | boolean | `false` |
| `selection.image_count` | integer、1以上 | `100` |
| `selection.scene_hint` | 空でないstring、任意 | なし |
| `selection.spoiler_sensitivity` | `low` / `medium` / `high` | `medium` |
| `selection.similarity_threshold` | number、0以上0.98以下 | `0.72` |
| `frame_extraction.heartbeat_interval_seconds` | number、0より大きい | `1.0` |
| `frame_extraction.scene_change_threshold` | number、0以上1以下 | `0.25` |
| `frame_extraction.scene_min_interval_seconds` | number、0より大きい | `0.5` |
| `frame_extraction.decode_backend` | `cpu` / `nvdec` | `cpu` |
| `frame_extraction.refinement_radius_seconds` | number、0以上 | `1.0` |
| `frame_extraction.max_frame_candidates` | integer、1以上3以下 | `3` |
| `candidate_moments.density_per_minute` | number、0より大きい | `2.0` |
| `context.language` | 空でないBCP 47相当のlanguage tag | `ja` |
| `context.subtitle_stream_index` | integer、0以上、任意 | 自動選択 |
| `context.audio_stream_index` | integer、0以上、任意 | 自動選択 |
| `ollama.host` | absolute HTTP(S) URL | `http://localhost:11434` |
| `ollama.timeout_seconds` | number、0より大きい | `60` |
| `ollama.max_parallel_requests` | integer、1以上 | `1` |
| `models.auto_upgrade` | boolean | `true` |
| `models.scene_catalog.name` | 空でないOllama tag | `qwen3-vl:8b-instruct` |
| `models.scene_catalog.num_ctx` | integer、32768以上 | `32768` |
| `models.candidate_annotation.name` | 空でないOllama tag | `qwen3-vl:8b-instruct` |
| `models.candidate_annotation.num_ctx` | integer、32768以上 | `32768` |
| `models.speech_to_text.name` | 空でないHugging Face repo ID | `dropbox-dash/faster-whisper-large-v3-turbo` |
| `models.speech_to_text.device` | faster-whisperが受理するdevice | `cuda` |
| `models.speech_to_text.compute_type` | CTranslate2が受理するcompute type | `float16` |
| `models.speech_to_text.beam_size` | integer、1以上 | `5` |
| `speech_to_text.vad_filter` | boolean | `true` |
| `speech_to_text.chunk_seconds` | number、0より大きい | `600` |
| `speech_to_text.overlap_seconds` | number、0以上かつchunk未満 | `5` |

`decode_backend = "nvdec"`、STT device、compute typeはsyntaxだけでなくpreflightの実能力検査にも合格する必要があります。word timestamp、Video Time mapping、chunk overlapの所有規則、Context Cue reliability policyは設定で無効化できないdomain contractです。

## モデルの役割

Scene CatalogとCandidate Annotationは独立したmodel設定とStage Fingerprintを持ちます。初期値は同じOllama modelですが、一方だけを変更できます。STTはfaster-whisper backendを使い、model名と実行profileを変更できます。どの役割も失敗時に別modelへ自動fallbackしません。

TOMLへ`expected_digest`、commit SHA、`revision`は書きません。実行開始時に各modelの完全なResolved Model Identityを取得し、そのrun内ではfreezeします。完全digestまたはcommit SHAはStage Fingerprint、cache manifest、`report.json`へ、短縮値は`report.md`へ記録します。

Resolved Model Identityが前回と同じならmodel依存cacheを再利用し、変わった場合はそのmodelに依存するStageだけを再計算します。frame抽出など無関係な上流cacheは再利用し、異なるidentityのCompleted Stageは共存させます。

## 自動upgrade

`models.auto_upgrade = true`が既定です。

- `true`: 処理Stage開始前に、distinctなOllama tagは`/api/pull`で一度だけ同期し、Hugging Face modelはremote `main`のcommitを解決してそのimmutable snapshotを取得します。Ollamaの省略tagと`:latest`は同じselectorとして重複排除します。
- `false`: 完全でload可能なlocal modelがあればnetworkへ更新確認せず使います。localにないmodelだけは自動downloadしてbootstrapします。
- 更新確認・downloadがoffline、timeout、registry障害、権限不足などで失敗しても、完全でload可能なlocal modelがあればwarningと`update_status = "unavailable"`を記録して継続します。local modelもなければfatalです。
- partial downloadは使用しません。`--reset-cache`はmodel storeを削除しません。

更新結果は設定名や実行identityと混ぜず、roleごとに次の`update_status`としてprovenanceへ記録します。

| Status | 意味 |
|---|---|
| `not_requested` | `auto_upgrade = false`で検証済みlocal artifactが使われた |
| `unchanged` | 同期後の完全identityが更新前と同じだった |
| `updated` | 同期後の完全identityが更新前から変わった |
| `bootstrapped` | 利用可能なlocal artifactがなく新たに取得された |
| `unavailable` | 同期前に利用可能なlocal artifactがあり、同期不能後の再検査に合格したlocal artifactがwarning付きで使われた |

`local_identity_before_update`は完全でrole capabilityを満たすlocal artifactだけに設定されます。存在していてもpartialまたはload不能なartifactはlocal fallbackとして扱いません。同期不能時はlocal storeを再解決・再検査し、同期中にtagやsnapshot stateが不完全になっていないことも確認します。`update_status`、更新時刻、`auto_upgrade`は現在runの`report.json`へ残す診断値であり、Completed Stage artifactや同じ実行identityのStage Fingerprintを変えません。

Ollamaにはremote digestだけを確認するdocumented read-only APIがないため、`auto_upgrade = true`は単なる通知ではなくpullによる同期です。更新時刻や`auto_upgrade`自体ではなく、実際にfreezeしたmodel identityがStage Fingerprintを変えます。

temperature 0でもmodel出力のbit-for-bit一致は保証しません。同じStage Fingerprintでは最初にschema/domain検証へ合格した結果をatomicにcacheし、以後は再samplingせず再利用します。`--reset-cache`後は意味的に同等でも表現の異なる結果になり得ます。
