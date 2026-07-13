# CLAUDE.md

## プロジェクト概要

`dtxt` — テキスト⇄構造化データのスキーマ中心双方向変換ライブラリ(Python)。

3つのコア機能:

1. **スキーマ推論** (`infer_schema`): テキスト集合からスキーマを生成する
2. **T2D** (`parse`): テキストをスキーマ準拠のオブジェクトに変換する
3. **D2T** (`render`): オブジェクトをテキストに変換する

差別化ポイントは「スキーマ推論 + 双方向変換 + round-trip 検証 (`parse(render(obj)) ≈ obj`)」の三点セット。round-trip テストはライブラリの一級機能として扱う。

- PyPI 名 / インポート名: `dtxt`(両者を一致させる。`dtx` は既存の別パッケージなので使わない)
- Python 3.10+
- src レイアウト

## アーキテクチャ

```
src/dtxt/
├── __init__.py       # 公開API: infer_schema, parse, render, configure, Schema
├── schema.py         # Schemaクラス
├── infer.py          # スキーマ推論パイプライン
├── t2d.py            # parse / parse_many
├── d2t.py            # render
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
- **機能単位でバックエンドを指定できる**ことが必須要件:

```python
dtxt.configure(
    infer=dtxt.backends.Anthropic("claude-sonnet-4-6"),
    parse=dtxt.backends.LlamaCpp("model.gguf", n_ctx=8192),
    render=dtxt.backends.Anthropic("claude-sonnet-4-6"),
)
```

- グローバル `configure` に加え、各関数の `backend=` 引数で個別上書きも可能にする

### llama.cpp バックエンド (backends/llamacpp.py)

- `llama-cpp-python` を使用。`response_format` に JSON Schema を渡して GBNF 制約デコーディングを利用する
- JSON Schema → GBNF で表現しきれない部分(`format`, `pattern`, 深い再帰)は**事後バリデーションに回す二段構え**にする。文法で保証できる範囲と事後検証の範囲を明確に分離すること
- インプロセスはシングルストリーム前提。バッチ処理は逐次 + プロンプトキャッシュ活用
- 小型ローカルモデルはプロンプト感度が高いので、プロンプトテンプレートをバックエンド側でオーバーライド可能にする

### API バックエンド (anthropic.py / openai.py)

- structured output / tool calling を優先利用。失敗時はリトライ + バリデーションループ
- バッチ処理(`parse_many`)は asyncio で並列化

### スキーマ推論 (infer.py)

- スキーマ発散対策として、**サンプリング + マージ方式**: 数件ずつ候補スキーマを生成し統合する。小さい n_ctx でも動作すること
- 「カバレッジ N% 以上のフィールドのみ採用」の閾値パラメータ(`min_coverage`)を持たせる

### T2D (t2d.py)

- 欠損・抽出失敗の表現を統一する: 抽出できなかったフィールドは `None`、「テキスト中に存在しないことが確認できた」と「見つけられなかった」を区別したい場合のためのオプションを検討(初期実装では `None` で統一してよい)
- `parse_many()` バッチ API を用意し、内部でバックエンドごとに最適化する

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

1. **M1**: コアインターフェース(`Schema` / `parse` / `render`)+ モックバックエンド + round-trip テストで縦に貫通
2. **M2**: Anthropic / OpenAI バックエンド実装(リトライ + バリデーションループ)
3. **M3**: llama.cpp バックエンド実装(GBNF 制約 + 事後バリデーション二段構え)
4. **M4**: `infer_schema`(サンプリング + マージ、`min_coverage`)
5. **M5**: `parse_many` バッチ最適化、D2T の文体制御、PyPI 公開(0.1.0)

※ 名前確保のため、M1 完了時点で動く骨格 + README を `0.0.1` として早めに PyPI に上げる(空プレースホルダーは PyPI 規約上削除対象になり得るため、最低限動くものにする)。

## 注意事項

- `dtx`(PyPI 上の既存パッケージ、AI Red Teaming ツール)とは無関係。ドキュメントやコードで混同しないこと
- プロンプトはコード内にハードコードせず `prompts/` またはモジュール定数に集約し、バックエンドごとのオーバーライドを可能にする
- 公開 API の破壊的変更は 0.x の間でも CHANGELOG に明記する

