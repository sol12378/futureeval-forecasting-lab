# FutureEval Forecasting Lab（Fall 2026 + MiniBench）

Metaculus公式テンプレート [metac-bot-template](https://github.com/Metaculus/metac-bot-template) のPublic Forkです。
FutureEval Fall 2026（project `33121` / `fall-futureeval-2026`）とMiniBench（`minibench`）に、
人手を介さない完全自律運用で予測と根拠コメントを提出します。

- 主予測：Claude Opus 5.5（`anthropic/claude-opus-5-5`）
- 構造化出力の解析：GPT-6.2 Sol（`openai/gpt-6.2-sol`、推論effort=low）
- 最新ニュース：AskNews（未設定の場合はリサーチなし。LLMに架空の「Web検索」をさせることはしない）
- APIキー：Claude・GPTとも、1本目のクレジットが切れたら2本目に自動で切り替え
- 実行基盤：GitHub Actions（Publicリポジトリのため無料枠）、20分間隔

## 構成

| ファイル | 役割 |
|---|---|
| `main.py` | 上流の `FallTemplateBot2026` を変更せずに継承した `GuardedBot`、実行モードの切り替え、提出後の検証 |
| `forecast_config.py` | モデル選択・予測回数・Secretsの存在確認（値はログに出さない） |
| `budget_guard.py` | プロバイダ別の月次予算ガードと、追記専用の費用台帳 |
| `scripts/ledger.sh` | `ops-ledger` ブランチの台帳を取得・追記する |
| `scripts/preflight.py` | 必須Secretsの有無を、高コストな処理の前に検査する |
| `.github/workflows/run_bot_on_tournament.yaml` | 本番（`ENABLE_LIVE_FORECASTS=true` のときだけ動く） |
| `.github/workflows/test_bot.yaml` | Bot Testing Area への手動テスト |
| `.github/workflows/ci.yaml` | Secretsを使わないオフライン検査（単体テスト・コンパイル） |

## 安全装置

0. **APIキーの切り替え（Claude・GPT共通）**：1本目が「クレジット切れ」を返したときだけ2本目に切り替えます。判定に使うのは、Claudeでは `billing_error` / credit balance is too low / `enforced_spend_limit_reached`、GPTでは `insufficient_quota` などです。レート制限や無効なキーでは切り替えず、失敗として通知します。月内は前回の実行が終わったキー枠から再開し、月が替わると1本目に戻ります。キー枠ごとの使用額は台帳の `cost_by_key_slot` に記録されます（キーの値は記録しません）。

1. **本番提出のスイッチ**：リポジトリ変数 `ENABLE_LIVE_FORECASTS` が `true` でない限り、本番ジョブはスキップされます。`main.py` 側でも同じ値を確認します（二重チェック）。
2. **重複提出の防止**（大会は1問につき1回のみ提出可）：Metaculus側の `already_forecasted` に加え、台帳に「提出済み」とある問題も除外します。両者が食い違った場合は提出せず、ジョブを失敗させて通知します（終了コード5）。
3. **提出漏れの検知**：実行後に大会を再取得し、対象にした問題がまだ予測なしで開いていれば失敗にします（終了コード4）。
4. **予算（キー1本ごとの上限）**：月の支出を、プロバイダ・キーごとに台帳で合計します（月はUTCで区切ります）。上限は1本あたりClaude $200・GPT $100で、2本あればClaude計$400・GPT計$200です。
   - 使用中のキーが自分の上限の90%に達したら、クレジット切れのエラーを待たずに次のキーへ切り替えます
   - 登録済みのキーがすべて90%に達したら、新しいLLM処理を開始しません（終了コード3）
   - 全キー合計の70%以上：予測回数を1回に減らします（節約モード）
   - 1回の実行の合計が `RUN_COST_CAP_USD`（既定$30）に達した時点で、以降の問題を止めます
   - 料金を計算できないモデルは起動時に拒否します。料金不明の呼び出しは1回$0.50として保守的に計上します
   - 台帳が読めない場合は$0とみなさず、停止します（フェイルクローズ）
5. **公開ログの保護**：本番モードでは、推論文と予測値をActionsログに出しません（コメントとしてMetaculusには投稿されます）。
6. **Metaculus Cup**：対象外のため、`ALLOW_METACULUS_CUP=true` を設定しない限り何もしません。

GitHub Actionsのジョブが失敗すると、リポジトリ所有者にメールで通知されます（GitHubの既定設定）。

## 本人による設定（Secrets / Variables）

`Settings → Secrets and variables → Actions`

| 種別 | 名前 | 内容 |
|---|---|---|
| Secret | `METACULUS_TOKEN` | Botアカウントのトークン |
| Secret | `ANTHROPIC_API_KEY` | Claude用（1本目。最初に使う） |
| Secret（任意） | `ANTHROPIC_API_KEY_2` | Claude用（2本目。1本目がクレジット切れになったときだけ使う） |
| Secret | `OPENAI_API_KEY` | GPT用（1本目。最初に使う） |
| Secret（任意） | `OPENAI_API_KEY_2` | GPT用（2本目。1本目がクレジット切れ `insufficient_quota` になったときだけ使う） |
| Secret（推奨） | `ASKNEWS_API_KEY` | AskNews（Metaculus公式手順の方式。Bot参加者は無料枠あり） |
| Secret（代替） | `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` | AskNewsのOAuth方式。`ASKNEWS_API_KEY` と両方ある場合はこちらが優先 |
| Variable | `ENABLE_LIVE_FORECASTS` | テストに合格してから `true` にする |
| Variable（任意） | `PREDICTIONS_PER_RESEARCH_REPORT` | 本番の予測回数。既定は3、最大5 |
| Variable（任意） | `CLAUDE_MONTHLY_BUDGET_PER_KEY_USD` / `GPT_MONTHLY_BUDGET_PER_KEY_USD` | キー1本あたりの月上限。既定は200 / 100 |
| Variable（任意） | `BUDGET_SOFT_RATIO` / `BUDGET_HARD_RATIO` / `RUN_COST_CAP_USD` | 既定は0.70 / 0.90 / 30 |
| Variable（任意） | `CLAUDE_MODEL` / `GPT_MODEL` | モデルの差し替え（litellmで料金を計算できるモデルに限る） |
| Variable（任意） | `GPT_REASONING_EFFORT` | GPTの推論量。既定 `low`（none/low/medium/high/xhigh/max） |

## AskNews無料枠の受け取り方（Metaculus公式Resourcesページより）

1. https://my.asknews.app で、**MetaculusのBotアカウントと同じメールアドレス**でアカウントを作成します。
2. AskNewsのDiscordで @freqai に連絡するか、contact@asknews.app にメールで以下を送ります：Bot名、AskNewsに登録したメールアドレス、氏名、LinkedInプロフィール、所属（会社・研究室・個人）。
3. AskNews側で無料枠（月1,000回、大会全体で4,000回、5M tokens）が有効化されたら、APIキーを発行して `ASKNEWS_API_KEY` に登録します。

- 最新ニュース（48時間以内）の検索は1回、過去記事の検索は5回として数えます。このBotは最新ニュースを1問1回だけ使います。
- アカウントは1人（1チーム）1つまでで、毎シーズン更新が必要です。大会以外の用途には使えません。

## 運用手順

1. Secretsを登録します。
2. `Actions → Test Bot → Run workflow` を実行し、Bot Testing Areaへの投稿をMetaculus上で確認します。
3. 合格したら `ENABLE_LIVE_FORECASTS=true` を設定し、`Forecast on new AI tournament questions` を手動で1回実行します。
4. 以降は20分ごとに自動で巡回します（GitHubのスケジュール実行には遅延があります。問題は約3時間開いているため許容範囲です）。

**緊急停止**：`ENABLE_LIVE_FORECASTS` を `false` にし、実行中のRunをキャンセルします。

## 資源配分計画（2026-11-21時点の見積もり。実測後に更新）

**出題量**（公式Resourcesページ）：本大会は4か月で300〜400問（月75〜100問）、MiniBenchは2週間ごとに約60問（月約120問）。
グループ問題の小問や条件付き問題を考慮し、**月250問**を上限として見積もります。

**1問あたりの費用（予測1回あたり）**
- Claude Opus 5.5（$4 / $20 per MTok）：入力3k〜6k、出力（思考を含む）2k〜6k tokens → 約$0.05〜0.15（中央値$0.10）
- GPT-6.2 Sol（$2 / $10 per MTok、effort=low）：構造化の検証2回 → 約$0.02

| 予測回数/問 | Claude 月額 | GPT 月額 |
|---|---|---|
| 1 | 約$25 | 約$5 |
| **3（初期設定）** | **約$75** | **約$15** |
| 5 | 約$125 | 約$25 |

| 資源 | 月の枠 | 配分 |
|---|---|---|
| Anthropic | 1本$200 × 2本 = $400 | 本番$75前後、テスト・開発$15、残りは予備。キー1が$180（90%）でキー2へ。合計$280（70%）で節約モード、キー2も$180で停止 |
| OpenAI | 1本$100 × 2本 = $200 | パーサ$15前後。初回投稿後にGPTの独立予測（アンサンブル）$30〜50、終了済み問題の評価$10〜15。キー1が$90でキー2へ。合計$140で節約、キー2も$90で停止 |
| AskNews | 前払いウォレット$15（今季のFutureEval向け無料枠は受付終了） | 従量課金：1検索 約$0.03〜0.09。累計150回で打ち止め（`ASKNEWS_TOTAL_CALL_CAP`）。それ以降と障害時はニュースなしで予測を続行 |
| GitHub Actions | Publicリポジトリのため無料 | 20分ごと、1回あたり約1〜3分 |

実測の費用は台帳（`ops-ledger` ブランチ）とActionsのStep Summaryで確認できます。

## 公開リポジトリの秘密・個人情報対策

| 層 | 仕組み | 防ぐもの |
|---|---|---|
| コミット時 | `.githooks/pre-commit` → `scripts/privacy_check.sh staged` | gitleaksで検出される秘密、`.env`・鍵ファイル、noreply以外の作者メール、ローカル拒否リストの個人情報 |
| push時 | `.githooks/pre-push` → 送信するコミット全件を再検査 | 同上（コミット済みの見落とし） |
| GitHub | Secret scanning + Push protection（有効化済み） | 既知形式のAPIキーを含むpushをサーバー側で拒否 |
| CI | `ci.yaml` の `privacy` ジョブ（gitleaksは版固定・sha256検証） | 新規コミットの秘密・作者メール・拒否リスト（Secret `PRIVACY_DENYLIST`）・ツリー全体 |
| `.gitignore` | `.env.*`、鍵、認証情報JSONなど | 誤って追加するのを防ぐ |

- 新しくcloneしたら `bash scripts/install_hooks.sh` を1回実行してください。
- 個人情報（メール、氏名、電話番号など）は、このPCの `~/.config/futureeval-forecasting-lab/denylist.txt` に1行ずつ書きます。このファイルはリポジトリの外にあり、コミットされません。CIでも同じ検査をする場合は、同じ内容をSecret `PRIVACY_DENYLIST` に登録します。検出しても値は表示しません。
- 台帳（`ops-ledger`）には費用・問題URL・状態だけを記録し、予測内容やエラー本文は記録しません。

## ルール上の注意

- 開催中の問題について、人間が予測を修正することは禁止されています（Botの出力を見てBotを修正することも含みます）。
- 予測には必ずBotのコメントを付けます（テンプレートが自動で投稿します）。
- 1人につきBotは1体まで。賞金の受け取りには、Botの説明またはコードの開示が必要です。
