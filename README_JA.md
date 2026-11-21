# FutureEval + MiniBench Launch Kit (2026-11-20)

## 重要：現時点の状態

これは **Metaculus公式テンプレートに重ねるパッチ** です。GitHubにリポジトリを自動作成したり、API Secretsを設定したものではありません。**実際の大会提出はまだ行われていません**。

- 公式テンプレート：https://github.com/Metaculus/metac-bot-template
- Metaculus参加：https://www.metaculus.com/futureeval/participate/
- 大会ルール：https://www.metaculus.com/notebooks/28751/futureeval-competition-rules/

## A. アカウントの準備（本人操作）

1. Metaculusで個人アカウントを作成。「My Forecasting Bots」で **Primary Botを1体**作り、`METACULUS_TOKEN` を控える。
2. 参加者フォームの必須項目を入力し、競技ルールを確認する。
3. Anthropic Console と OpenAI Platform にアクセス。毎月の支給クレジットが **通常のAPI呼び出しに利用できる** ことを確認する。クレジットの失効日、対象プロジェクト、追加課金の設定を確認する。
4. APIキーを発行する。**キーをChatGPTやリポジトリ、Issue、Actionsログに貼らない。**
5. できればAskNewsも登録し、`ASKNEWS_CLIENT_ID` と `ASKNEWS_SECRET` を取得する。AskNewsなしではリアルタイム検索なしで動作するため、予測精度への悪影響を想定する。

## B. GitHubのForkを作る（本人操作）

1. https://github.com/Metaculus/metac-bot-template にアクセスして **Fork** を押す。
2. Owner=`sol12378`、リポジトリ名は例 `futureeval-forecasting-lab`、**Public** として作成する。
3. 自分のPCでクローンする（以下は名前を上記とした例）。

```bash
git clone https://github.com/sol12378/futureeval-forecasting-lab.git
cd futureeval-forecasting-lab
```

4. このZIPの内容を **forkのルート**に展開する。`main.py` は元の公式版が存在するままにする。
5. 次のコマンドを実行する。

```bash
python3 scripts/prepare_fork.py
python3 -m unittest discover -s tests -v
python3 -m compileall -q main.py forecast_config.py scripts

git add main.py forecast_config.py scripts .github/workflows README_JA.md tests .env.example.launchkit
git commit -m "Configure Claude/GPT FutureEval and MiniBench dual-tournament bot"
git push origin main
```

**注意**：公式テンプレートの更新で`prepare_fork.py`が停止した場合は無理に続行しない。差分を確認して再適用する。

## C. GitHub Secretsを登録（本人操作）

Forkの `Settings → Secrets and variables → Actions → Secrets → New repository secret` に以下を登録する。

| Name | 内容 |
|---|---|
| `METACULUS_TOKEN` | MetaculusのBotアカウント用トークン |
| `ANTHROPIC_API_KEY` | Claude用APIキー |
| `OPENAI_API_KEY` | GPT用APIキー |
| `ASKNEWS_CLIENT_ID` | 任意・AskNews |
| `ASKNEWS_SECRET` | 任意・AskNews |

`Settings → Secrets and variables → Actions → Variables` の `ENABLE_LIVE_FORECASTS` は **最初は作らないか `false`** にする。`CLAUDE_MODEL`、`GPT_MODEL` は未設定であればコード内の既定モデルを使用する。

**重要**：OpenAI/Anthropicのアカウント画面で、自動リチャージや過剰請求につながる設定を確認。無料クレジットだけを使う運用なら追加支出しないよう予算・アラートを設定する。このキットはAPIプロバイダーの残高を取得できないため、月額$200/$100の**厳密な金額制限は自動では保証できない**。

## D. テスト → 本番

1. `Actions`タブでWorkflowを有効化する（Forkした場合は手動承認が必要なことがある）。
2. `Test Bot` → `Run workflow`。`Bot Testing Area` への予測テスト。実際にBotアカウントへ切り替え、テスト予測の反映を確認。
3. テストが成功した場合のみ `Variables → ENABLE_LIVE_FORECASTS = true` を登録する。
4. `Forecast on new AI tournament questions`を**手動で1回実行**し、ForecastのログがFall 2026とMiniBenchの両方を対象にしていることを確認する（その瞬間に新規問題がなければ `No new questions` は正常）。
5. その後は20分おきに自動実行する。スケジュールはGitHub Actionsの混雑等による遅延・スキップがあり得る。
6. テスト実行の終了ログとBotアカウントの実際の投稿を照合する。「Run success」だけでは予測提出成功を意味しない。

## E. 初期の費用・研究方針

- **初期版**：Claude Sonnet 5を主予測に使用、GPT-5.4 miniを出力解析・要約に使用。各問題のリサーチ1回・予測1回。
- **注意**：これは独立したClaude/GPTアンサンブルではない。両モデルの独立予測比較は、安定稼働後の別フェーズで実装する。
- AskNews未設定：研究取得なし。最新ニュースに乏しいためスコア評価に注意。
- 利用枠：Claude $200/月、GPT $100/月はAPI提供元のクレジット条件が一致する場合の **上限目安** であり、この設定がクレジット残量を管理するわけではない。
- 初回は `RESEARCH_REPORTS_PER_QUESTION=1`、`PREDICTIONS_PER_RESEARCH_REPORT=1` でコスト最小化。連続運用で利用費を実測してから増やす。

## F. トラブルシュート

| エラー | まず確認すること |
|---|---|
| `SETUP_ERROR` | GitHub Actions Secretsの名前、APIキーの対象プロジェクト |
| `Model not found` | Anthropic/OpenAIのモデル利用権限、VariablesのモデルID |
| `429` | APIレート制限。必要なら予測数と同時実行数を下げる |
| `Payment required / insufficient quota` | クレジットがAPIに使えるか、課金・残高設定 |
| 成功なのに提出がない | 新規対象問題が0件か、Botアカウントで確認していない可能性 |
| `No new questions` | 本大会とMiniBenchの出題タイミング。正常な場合あり |
| 予測が遅い | GitHub Actionsのスケジュール遅延、単一ジョブの長時間化 |

## G. 本番停止方法

緊急停止は `Settings → Secrets and variables → Actions → Variables` の `ENABLE_LIVE_FORECASTS` を `false` に変更する。すでに開始済みのRunは `Actions` から手動キャンセルする。停止後、必要に応じてAPIキーを失効する。

## H. 今後の実装（初回提出後）

- 実際の出題でClaudeとGPTの独立推論を併走させ、妥当なアンサンブルを構築する。
- 終了済み問題に対するバックテストと校正、時点情報のリーク防止を実装する。
- 提出率・API費用・予測精度のメトリクスを管理する。
- 2026年12月18日以降、ACPとCloudflare/WSL/acpnode1への統合を検討する。

## セキュリティ・ルール

開催中の問題への人間の予測修正は禁止されている。終了済み問題の研究を本番ボットへ反映させる。Publicリポジトリには `.env`、APIキー、Botトークン、未公開の秘密情報を絶対にコミットしない。Workflowは `permissions: contents: read` のみ。予測アルゴリズムは公開される。

### 対象外大会の無効化

この起動キットは、公式テンプレートにある `run_bot_on_metaculus_cup.yaml` の定期実行も無効化します。Cupへ意図せず予測を提出したりAPI費用を消費したりしないためです。
