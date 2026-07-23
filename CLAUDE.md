# CLAUDE.md

## プロジェクト概要

`dtxt` — テキスト⇄構造化データのスキーマ中心双方向変換ライブラリ(Python)。

3つのコア機能:

1. **スキーマ推論** (`SchemaInferer`): テキスト集合からスキーマを生成する
2. **T2D** (`StructuredEntityExtractor`): テキストをスキーマ準拠のオブジェクトに変換する
3. **D2T** (`StructuredEntityRenderer`): オブジェクトをテキストに変換する

差別化ポイントは「スキーマ推論 + 双方向変換 + round-trip 検証 (`extractor.extract(renderer.render(obj)) ≈ obj`)」の三点セット。round-trip テストはライブラリの一級機能として扱う。

公開APIはクラスベース(`backend` はコンストラクタ引数で渡す。グローバル `configure()` は廃止した)。

- PyPI 名 / インポート名: `dtxt`(両者を一致させる。`dtx` は既存の別パッケージなので使わない)
- Python 3.10+
- src レイアウト

## アーキテクチャ

```
src/dtxt/
├── __init__.py       # 公開API: SchemaInferer, StructuredEntityExtractor,
│                     #          StructuredEntityRenderer, check_roundtrip, Schema
├── schema.py         # Schemaクラス
├── entities.py       # schema-free抽出(Flat/NestedEntityExtractor)+ 型名正規化
│                     #  (EntityTypeNormalizer)-- infer.py が内部で使う実装基盤
├── infer.py          # SchemaInferer: entities.py を使ったスキーマ推論パイプライン
├── t2d.py            # StructuredEntityExtractor: parse相当(schema指定あり)
├── d2t.py            # StructuredEntityRenderer: render相当(schema指定あり)
├── roundtrip.py      # round-trip検証
└── backends/
    ├── base.py       # Backend Protocol + capabilities
    ├── anthropic.py
    ├── openai.py
    └── llamacpp.py
```

### Schema (schema.py)

- 内部表現は **JSON Schema 互換**。Pydantic モデルからの生成・への変換もサポートする
- D2T 用の記述メタデータ(description、例文、文体ヒント)を JSON Schema の拡張キーワード(`x-dtxt-*` プレフィックス)として持たせる
- バリデーションは JSON Schema ベースで行う

### Backend 抽象化 (backends/base.py)

```python
class Backend(Protocol):
    def generate(self, prompt: str, *, schema: dict | None = None) -> str: ...
    @property
    def capabilities(self) -> set[str]: ...
        # 例: {"constrained_decoding", "json_mode", "tool_calling"}
```

- 上位層は `capabilities` を見て実行パスを選ぶ:
  - `constrained_decoding` あり(llama.cpp + GBNF)→ 構文検証は省略可、意味的バリデーションのみ
  - なし(API系)→ リトライ + バリデーションループで担保
- **機能単位でバックエンドを指定できる**ことが必須要件。`backend` は各クラスのコンストラクタ引数として渡す(グローバル設定は持たない):

```python
extractor = dtxt.StructuredEntityExtractor(
    dtxt.backends.LlamaCpp("model.gguf", n_ctx=8192), schema
)
renderer = dtxt.StructuredEntityRenderer(dtxt.backends.Anthropic("claude-sonnet-4-6"), schema)
inferer = dtxt.SchemaInferer(dtxt.backends.Anthropic("claude-sonnet-4-6"))
```

### llama.cpp バックエンド (backends/llamacpp.py)

- `llama-cpp-python` を使用。`response_format` に JSON Schema を渡して GBNF 制約デコーディングを利用する
- JSON Schema → GBNF で表現しきれない部分(`format`, `pattern`, 深い再帰)は**事後バリデーションに回す二段構え**にする。文法で保証できる範囲と事後検証の範囲を明確に分離すること
- インプロセスはシングルストリーム前提。バッチ処理は逐次 + プロンプトキャッシュ活用
- 小型ローカルモデルはプロンプト感度が高いので、プロンプトテンプレートをバックエンド側でオーバーライド可能にする

### API バックエンド (anthropic.py / openai.py)

- structured output / tool calling を優先利用。失敗時はリトライ + バリデーションループ
- バッチ処理(`StructuredEntityExtractor.extract_many`)は asyncio で並列化

### entities.py -- schema-free 抽出 + 型名正規化

`SchemaInferer` の実装基盤。人間がテキスト集合からスキーマを設計するときの「まずどんな情報が含まれているか schema-free に見る → 名寄せする → 構造を確定させる」という手順をなぞる:

- `FlatEntityExtractor` / `NestedEntityExtractor`: 1テキストを `(type, value)` のフラットな(または `children` を持つグループ込みの)エンティティ列に、schema-free で抽出する。同じ `type` の繰り返しはそれ自体が配列フィールドのシグナルなので潰さない
  - `NestedEntityExtractor(backend, max_depth=1)`: `children` がさらに `children` を持てる深さの上限を `max_depth` で指定する(デフォルト1)。上限を超えるネストは出力 JSON Schema・レスポンスのパース両方で切り捨てる。将来的に上限を緩める方向で検討中だが、無制限の再帰は GBNF 制約デコーディングと相性が悪い(後述)ため、当面は「設定可能な有限値」に留める
- `EntityTypeNormalizer`: 観測された型名をコーパス横断で正規化する。**1階層ずつ**処理する: まずその階層の型名を(例文付きで)1回の backend 呼び出しでマージし、次にその正準グループ型の `children` を(コーパス中の全出現から)プールして、1階層下を再帰的に正規化する。トップレベルを先に確定させてから子をプールすることで、子階層のマージに十分なサンプル数を確保する
  - 同じ正準型がコーパス中で scalar(leaf)としても group(children持ち)としても出現するケースを、このクラス自身は解決しない(型名の正規化のみが責務)。scalar/object のどちらとして扱うかはコーパス全体のカバレッジを持つ `SchemaInferer` 側のマージ処理に委ねる

### スキーマ推論 (infer.py: SchemaInferer)

`SchemaInferer.infer(texts)` は次のパイプラインで動く:

1. `NestedEntityExtractor` で各テキストを schema-free に抽出する(1テキスト1回のbackend呼び出し。抽出に失敗したテキストはスキップし、全滅した場合のみ `InferError`)
2. `EntityTypeNormalizer` でコーパス横断の型名を正規化する(`fit` → `transform`)
3. 正規化後のエンティティ列を、階層ごとに再帰的にマージしてスキーマを組み立てる:
   - 「カバレッジ N% 以上のフィールドのみ採用」の閾値パラメータ(`min_coverage`)は各階層で使う、再帰の停止・採否基準
   - 1インスタンス内での型の繰り返しは配列フィールドのシグナル(`FlatEntityExtractor` と同じ考え方)
   - 同じ正準型の出現のうち過半数が group(`children` あり)なら object フィールドとして扱い、その `children` を再帰的にマージする。過半数が scalar なら文字列フィールドとして扱う(leaf/group 両方で出現するケースの解決はここで行う)

### T2D (t2d.py: StructuredEntityExtractor)

- 欠損・抽出失敗の表現を統一する: 抽出できなかったフィールドは `None`、「テキスト中に存在しないことが確認できた」と「見つけられなかった」を区別したい場合のためのオプションを検討(初期実装では `None` で統一してよい)
- `extract_many()` バッチ API を用意し、内部でバックエンドごとに最適化する

## 依存関係の方針

- **コアは pydantic + jsonschema のみ**。バックエンドは全て optional extras + 遅延インポート
- `llama-cpp-python` はビルドが重いため、絶対にコア依存に入れない

```toml
[project.optional-dependencies]
openai = ["openai>=1.0"]
anthropic = ["anthropic>=0.40"]
llamacpp = ["llama-cpp-python>=0.3"]
all = ["dtxt[openai,anthropic,llamacpp]"]
```

- バックエンド未インストール時は、インストールコマンドを含む明確なエラーメッセージを出す

## 開発規約

- パッケージ管理: `uv`
- Lint / Format: `ruff`(`ruff check` + `ruff format`)
- 型チェック: `mypy --strict`(公開APIは完全に型付けする)
- テスト: `pytest`
  - LLM 呼び出しはモックバックエンド(`backends/mock.py` を作る)でテストする。実 LLM を叩くテストは `-m integration` マーカーで分離し、CI のデフォルトでは走らせない
  - round-trip 検証のテストを重点的に書く
- コミットは Conventional Commits(`feat:`, `fix:`, `docs:` など)

## よく使うコマンド

```bash
uv sync --all-extras          # 開発環境セットアップ(llamacppを除く場合は --extra openai --extra anthropic)
uv run pytest                 # ユニットテスト
uv run pytest -m integration  # 実LLMを使う統合テスト
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
```

## マイルストーン

1. **M1**: コアインターフェース(`Schema` / T2D / D2T)+ モックバックエンド + round-trip テストで縦に貫通
2. **M2**: Anthropic / OpenAI バックエンド実装(リトライ + バリデーションループ)
3. **M3**: llama.cpp バックエンド実装(GBNF 制約 + 事後バリデーション二段構え)
4. **M4**: スキーマ推論(サンプリング + マージ、`min_coverage`)-- 当初は候補スキーマJSONを直接生成する方式だったが、後に entities.py ベースの `SchemaInferer`(schema-free 抽出 + 型名正規化 + カバレッジベースの再帰マージ)に置き換えた(M6)
5. **M5**: バッチ最適化、D2T の文体制御、PyPI 公開(0.1.0)
6. **M6**: 公開APIをクラスベースに刷新(`parse`/`render`/`infer_schema` 関数 → `StructuredEntityExtractor`/`StructuredEntityRenderer`/`SchemaInferer`、グローバル `configure()` 廃止)。`SchemaInferer` を entities.py(`Flat/NestedEntityExtractor` + `EntityTypeNormalizer`)を使った schema-free 抽出+再帰マージ方式に置き換え。`NestedEntityExtractor` のネスト上限を `max_depth` として設定可能にし、`EntityTypeNormalizer` を多階層再帰対応に一般化

※ 名前確保のため、M1 完了時点で動く骨格 + README を `0.0.1` として早めに PyPI に上げる(空プレースホルダーは PyPI 規約上削除対象になり得るため、最低限動くものにする)。

## 注意事項

- `dtx`(PyPI 上の既存パッケージ、AI Red Teaming ツール)とは無関係。ドキュメントやコードで混同しないこと
- プロンプトはコード内にハードコードせず `prompts/` またはモジュール定数に集約し、バックエンドごとのオーバーライドを可能にする
- 公開 API の破壊的変更は 0.x の間でも CHANGELOG に明記する

