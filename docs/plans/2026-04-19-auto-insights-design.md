# Auto Insights — Design

## Goal

`openkb add` 완료 후 쿨다운 기반으로 백그라운드에서 인사이트를 자동 생성한다.

## Architecture

```
openkb add doc.md
  → add 완료 후 _maybe_trigger_insights() 호출
  → .openkb/last_insights.json 읽기 → 쿨다운 만료?
    ↓ 예                        ↓ 아니오
    threading.Thread(           → 종료 (skip)
      target=_bg_insights,
      daemon=True
    ).start()
    → _bg_insights():
        generate_insights(graph)
        → .openkb/insights.md 저장
        → .openkb/last_insights.json 갱신
        → 터미널에 3-5줄 요약 출력
    → atexit.register(_wait_for_insights)
        → 프로세스 종료 시 thread 완료 대기 (최대 10초)
```

## State File

`.openkb/last_insights.json`:

```json
{
  "last_run": "2026-04-19T10:30:00Z",
  "cooldown_seconds": 3600
}
```

`cooldown_seconds`는 `openkb config set insights_cooldown <seconds>`로 변경 가능. 기본값 3600 (1시간).

## Terminal Output

쿨다운 만료 시 (백그라운드에서 인사이트 생성):

```
✓ Added 3 documents. Insights refreshing in background...
  Communities: 5 | Orphans: 2 | Surprising: 1
  Full report: .openkb/insights.md
```

쿨다운 미만료 시:

```
✓ Added 3 documents. (Insights cached 23m ago — run `openkb insights` for fresh)
```

## New File

`openkb/graph/insights_bg.py` — 백그라운드 인사이트 생성 로직:

- `maybe_trigger_insights(kb_dir)`: 쿨다운 체크 후 조건부 thread spawn
- `_bg_insights(kb_dir)`: 실제 인사이트 생성 + 파일 저장 + 요약 출력
- `_wait_for_insights()`: atexit 핸들러, thread 완료 대기 (최대 10초)

## Changed Files

| File | Change |
|------|--------|
| `openkb/cli.py` | `add` 명령 완료 후 `maybe_trigger_insights()` 호출 |
| `openkb/graph/insights_bg.py` | NEW — 백그라운드 인사이트 생성 로직 |
| `openkb/config.py` | `insights_cooldown` 설정 키 추가 |

## Validation

- `openkb add` 실행 후 쿨다운 만료 시 `.openkb/insights.md` 파일 생성 확인
- `openkb add` 연속 실행 시 첫 번째만 인사이트 생성, 이후는 캐시 메시지 출력
- `openkb config set insights_cooldown 60` 후 1분 뒤 재실행 시 인사이트 재생성
- 프로세스 종료 시 백그라운드 thread가 정상 완료 (insights.md 파일이 손상 없이 저장)