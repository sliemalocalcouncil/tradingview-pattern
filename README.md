# Stock Pattern Scanner v2 — Explainable Scoring Engine

US 주식 차트 패턴 (총 9개) 을 스캔하고 **설명 가능한 9-차원 점수**로 평가한 뒤, Telegram 으로 알림을 보내는 GitHub Actions 기반 봇입니다.

## v2.2.1 — Starter 플랜 기본화

- `POLYGON_REQ_PER_MIN` 기본값 `5` → `0` (Starter+ 무제한 가정)
- 자동 스캔 cron `매시 :37` → `매 30분 (:00, :30)` (Starter 속도 활용)
- `DAILY_LOOKBACK_DAYS` 기본값 `320` → `500` (Starter 의 5년치 데이터로 long-base 패턴 정확도 ↑)
- `MIN_DOLLAR_VOLUME` 기본값 `$5M` → `$10M` (Starter 사용자가 다루는 종목군 가정)
- README 에 플랜별 운용 가이드 별도 섹션
- Free tier 는 명시적 Variable 설정 시에만 동작

## v2.2.0 — 운영 안정성 묶음

리뷰팀 P0/P1 지적사항 전부 반영:

- **Prior_state 연결** (P0-1): `_state.json` 의 이전 state 가 state machine 에 실제로 흘러들어가 `breakout_failed`, `retest_hold` 분기가 동작.
- **Active signal tracker** (P0-2): detector 가 더 이상 매칭하지 않아도 invalid_below / trigger 를 5 거래일간 추적해 `INVALIDATED` / `BREAKOUT_FAILED` 알림 발사.
- **ET 정렬 60분봉** (P0-3): 30분봉을 받아 ET 09:30/10:30/.../15:30 으로 직접 resample. 기존 native 1시간봉은 `HOURLY_ALIGNMENT_MODE=native` 로 fallback 가능.
- **거래일 캘린더 기반 forward return** (P0-4): NYSE 캘린더로 ret_eod / ret_1d/3d/5d 정확 계산. max_dd_5d / max_fav_5d / touched_invalid_5d 도 실제 계산.
- **Volume confirmed breakout** (P1): V3_breakout_bar 또는 V5_bo_expansion ≥ 60 인 경우에만 `BREAKOUT_CONFIRMED` 승격.
- **RETEST_HOLD 알림 활성화** (P1): `retest` alert_type 추가, 🔵 이모지.
- **ENABLED_PATTERNS 토글** (P1): 신규 detector 노이즈 격리용 환경변수.
- **All-signal state save** (P1): best 만 알림하되, 모든 signal 의 state 는 저장 → 다음 cycle 의 prior_state 가 정확.
- **`latest_top_signals.csv` stale 방지** (P1): signal 0건일 때도 빈 파일로 갱신.
- **CI smoke test** (P1): `.github/workflows/smoke.yml` — push/PR 마다 4개 테스트 스위트 자동 실행.
- **Atomic state save** (P2): 임시 파일 + os.replace 로 부분 손상 방지.
- **Schema version** (P2): `features_json["_schema_version"] = "1.0"` 저장.

설계 문서는 `docs/v2_design.md` (별도) 를 참조하세요.

- **9개 패턴 detector** (검출과 점수 계산을 분리)
  - 초기 v1: Double Bottom · Bull Flag (HTP 변형 포함) · VCP · Ascending Triangle
  - v1.5 추가: Cup with Handle · Inverse Head & Shoulders · Base-on-Base · Tight Consolidation · Breakout Retest Hold
- 9차원 공통 점수 (T/G/VC/SR/V/BR/R/M/L) + 패턴별 가중치 매트릭스
- Veto 규칙 (유동성/리스크 명확성/실적/시장 regime)
- 패턴 상태 머신 (`absent → forming → candidate → setup → breakout_confirmed → ...`)
- `features_json` 으로 sub-feature 전부 CSV 에 보존
- Forward-return tracker (T+1h, EOD, T+1d/3d/5d)
- 60분봉 세션 경계 버그 수정 (`design.md` C5)
- Volume 분해 (max() 1개 값 → 5개 sub-feature)

> **주의**: v1.5 의 신규 5개 detector 는 합성 데이터 기반 smoke test 만 통과한 상태입니다. 실제 시장 데이터에서의 calibration 은 forward-return 200건+ 누적 후 v2 에서 진행 예정입니다. 처음 1–2주는 알림 임계값을 보수적으로 설정 (`MIN_ALERT_SCORE=75` 권장) 하시고 후속 분석을 권장합니다.

---

## 지원 패턴 (9개)

| 패턴 | 타임프레임 | 트리거 | invalid_below | 비고 |
|------|-----------|--------|---------------|------|
| Double Bottom | hourly | neckline (두 저점 사이 최고가) | min(L, R) × 0.985 | 반전 |
| Bull Flag | hourly | pole peak | flag_low × 0.985 | 연속 |
| High Tight Pullback | hourly | pole peak | flag_low × 0.985 | 강한 연속 (Bull Flag 변형) |
| VCP | daily | 마지막 pivot high | 마지막 low × 0.985 | 연속 (O'Neil/Minervini) |
| Ascending Triangle | hourly | 평탄한 상단 | 가장 최근 low × 0.985 | 연속/돌파 |
| Cup with Handle | daily | 우측 rim | handle_low × 0.985 | 연속 (O'Neil) |
| Inverse Head & Shoulders | hourly | neckline (두 어깨 사이 최고) | head × 0.985 | 반전 |
| Base-on-Base | daily | 두 번째 base 의 우측 rim | 두 번째 base low × 0.985 | 강한 연속 |
| Tight Consolidation | daily | 최근 N봉 최고가 | 최근 N봉 최저가 × 0.985 | 압축/돌파 |
| Breakout Retest Hold | hourly | 재돌파 trigger (= 원 level) | retest_low × 0.99 | 2차 진입 |

---

## 빠른 시작 (3 단계)

### 1) GitHub repo 시크릿 등록
저장소 **Settings → Secrets and variables → Actions**:

| Secret | 필수 | 설명 |
|--------|------|------|
| `POLYGON_API_KEY` | ✅ | https://polygon.io 의 API key. **Starter ($29/월) 이상 권장**, Free tier 도 동작 |
| `TELEGRAM_BOT_TOKEN` | (알림 사용 시) | https://t.me/BotFather 에서 봇 생성 후 받은 토큰 |
| `TELEGRAM_CHAT_ID` | (알림 사용 시) | 본인 채팅 ID (개인 채팅은 양수, 그룹은 음수) |

(선택) **Variables** 로 동작 튜닝 가능. **기본값은 Polygon Starter+ 기준**:

| Variable | 기본값 (Starter) | Free 시 권장값 | 의미 |
|----------|-----------------|---------------|------|
| `POLYGON_REQ_PER_MIN` | `0` (무제한) | `5` | 분당 요청 제한. Free 면 반드시 `5` |
| `MIN_CANDIDATE_SCORE` | `60` | `60` | candidate state 진입 임계값 |
| `MIN_ALERT_SCORE` | `70` | `70` | 알림 발송 최저 점수 (보수적으로 시작하려면 `75`) |
| `MIN_DOLLAR_VOLUME` | `10000000` | `5000000` | 일평균 거래대금 floor (USD) |
| `DAILY_LOOKBACK_DAYS` | `500` | `320` | 일봉 lookback (Starter 5년 데이터 활용) |
| `ENABLED_PATTERNS` | `all` | `all` | 콤마 구분 (예: `double_bottom,vcp`) — 신규 detector 노이즈 격리용 |
| `HOURLY_ALIGNMENT_MODE` | `et_aligned` | `et_aligned` | `et_aligned` (기본) 또는 `native` (legacy, 30분 skew 버그 있음) |
| `ACTIVE_MONITORING_TRADING_DAYS` | `5` | `5` | 알림 후 active monitor 추적 기간 |

> **Free → Starter 업그레이드 시 변경할 것**: `POLYGON_REQ_PER_MIN=0` 설정하시면 됩니다. 위에 적힌 다른 기본값(`MIN_DOLLAR_VOLUME=10M`, `DAILY_LOOKBACK_DAYS=500`)은 Starter 가정 하에 코드에 박혀있어 Variable 안 만들어도 자동 적용됩니다.

### 2) `ticker.txt` 작성
한 줄에 한 종목씩 (대소문자 무관, `#` 으로 시작하는 줄은 주석):

```
AAPL
NVDA
TSLA
# 적당한 유동성 (일평균 거래대금 $5M+) 종목이 적합
```

### 3) 워크플로우 실행
- **수동**: Actions 탭 → "Stock Pattern Scan" → "Run workflow" 버튼
- **자동**: 미국 정규장 시간(UTC 14–20시, 월–금) **30분 간격** (:00, :30) 자동 실행

결과는 `result/` 폴더에 자동 커밋됩니다.

---

## 결과 파일 구조

```
result/
├── _state.json                       # 알림 dedup 상태 (cooldown 체크)
├── latest_run_summary.json           # 최근 실행 요약
├── latest_top_signals.csv            # 최근 실행 상위 50개
├── scan_<timestamp>.csv              # 실행별 스냅샷
├── run_summary_<timestamp>.json      # 실행별 요약
└── signals_log.csv                   # 누적 로그 (forward returns 추적용)
```

`signals_log.csv` 의 41개 컬럼:
- 식별자: `run_id, asof, ticker, pattern, pattern_state, final_score`
- 9차원 점수: `trend, geometry, compression, sr_quality, volume, readiness, risk, market, liquidity`
- 가중 기여도: `comp_trend, comp_geometry, …, comp_liquidity` (합 = `final_score`)
- 가격 정보: `price, trigger, invalid_below, measured_move_target`
- Veto: `veto_triggered, veto_reasons`
- 설명 blob: `features_json` (47개 sub-feature 모두 JSON 으로)
- 검증용: `ret_1h, ret_eod, ret_1d, ret_3d, ret_5d, max_dd_5d, max_fav_5d, touched_invalid_5d`
- 분석 태깅: `post_hoc_tag` (수동 태깅용)

---

## 로컬 실행 (디버깅)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export POLYGON_API_KEY=xxx
export TELEGRAM_BOT_TOKEN=xxx       # 또는 ↓
export DRY_RUN=true                  # 알림 발송 없이 동작 확인

python scripts/scan.py
```

API 키 없이 동작 확인만 하고 싶다면:
```bash
python tests/smoke_test.py   # 합성 데이터로 전 파이프라인 1회 실행
```

---

## 알림 메시지 예시

```
🟡 SETUP — `NVDA`
_Vcp_  ·  state: `setup`
score: *82.4*  (G:78 T:88 V:71 SR:80 BR:74)
price: `$925.10`
trigger: `$932.50`
invalid: `$905.20` (_2.1% stop_)
target: `$960.00`
```

알림 종류:
- 👀 `watch` — candidate (점수 ≥ MIN_ALERT_SCORE 이지만 trigger 까지 거리 큼)
- 🟡 `setup` — trigger 까지 ≤ 0.5 ATR 진입 임박
- 🟢 `breakout` — 종가가 trigger 돌파 + **거래량 확장 (V3 또는 V5 ≥ 60)**
- 🔵 `retest` — 돌파 후 trigger 까지 되돌림 후 hold (= 2차 진입 신호)
- 🔴 `failed` — 돌파 후 trigger 하향 이탈 (active monitor 가 detector 미매칭에서도 잡음)
- ⚫ `invalidated` — invalid_below 하향 이탈 (active monitor 가 detector 미매칭에서도 잡음)

같은 종목/패턴은 6시간 cooldown. 단, **점수 8점 이상 상승**하거나 state 가 바뀌면 즉시 재발송.

---

## Polygon 플랜별 운용 가이드

본 스캐너는 종목당 평균 **2 API 요청** (daily + intraday) 를 씁니다.

### Starter ($29/월) — 기본 가정

- 분당 무제한 요청
- 5년치 historical data
- 종목당 약 1초 처리, **30종목 기준 1분, 200종목 기준 4분**
- **GitHub Actions 30분 cron + 200종목까지 운용 가능**
- 추가 설정 불필요 (코드 기본값이 Starter 기준)

### Developer ($79/월)

- Starter 와 동일한 처리 속도 (이미 무제한이므로 추가 이득 없음)
- 차이점: 5분 지연 → 실시간 (15분 → 0분)
- Starter 로 충분, 실시간 절실하면 업그레이드

### Free ($0)

- 분당 5 요청 제한 → **종목당 ~12초**
- **30종목 = 약 6–12분, 그 이상은 GH Actions timeout (25분) 위험**
- 운용 시 반드시 Variable 추가:
  - `POLYGON_REQ_PER_MIN` = `5`
  - `MIN_DOLLAR_VOLUME` = `5000000` (10M 기본값은 너무 빡셈)
  - 또한 watchlist 를 30종목 이하로 유지

---

## 다음 단계 (v2/v3 로드맵 — 설계서 §8 참조)

- v1.5 ✅: 9개 패턴 모두 구현 (이 버전)
- v2 (P0 작업): 상태 머신 prior_state 연결 / active signal tracker / 60분봉 정렬 재설계 / forward return 거래일 기준 재작성
- v2 (P1 작업): volume-confirmed breakout / earnings calendar 자동 조회 / sector ETF RS / universe percentile RS / GitHub Actions smoke test job
- v3: 3개월 forward-return 데이터 확보 후 캘리브레이션 → 패턴별 임계값 자동 조정
- v3: LightGBM re-ranker (`features_json` sub-feature 입력)

---

## 라이선스

코드는 자유롭게 사용/수정/배포 가능합니다 (MIT). 단, 본 봇이 만들어내는 알림은 매매 권유가 아니며, 실거래 손실에 대한 책임은 사용자 본인에게 있습니다.
